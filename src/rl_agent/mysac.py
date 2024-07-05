import itertools
import os

import pandas as pd
from torch.nn import functional as F
import torch as th
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import polyak_update
from torch.optim import AdamW


class MySAC(SAC):

    def __init__(
            self,
            n_priming_steps: int = 1,
            reset_interval=None,
            reset_critic_layers=None,
            reset_actor_layers=None,
            weight_decay_actor: float = 0,
            weight_decay_critic: float = 0,
            priming_weight_decay_actor: float = 0,
            priming_weight_decay_critic: float = 0,
            weight_dir: str = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            *args,
            **kwargs
        )
        self.n_priming_steps = n_priming_steps
        self.reset_critic_layers = [] if reset_critic_layers is None else reset_critic_layers
        self.reset_actor_layers = [] if reset_actor_layers is None else reset_actor_layers
        self.reset_interval = reset_interval

        # Regularization
        self.priming_weight_decay_actor = priming_weight_decay_actor
        self.weight_decay_actor = weight_decay_actor
        self.priming_weight_decay_critic = priming_weight_decay_critic
        self.weight_decay_critic = weight_decay_critic

        if weight_dir is not None:
            os.makedirs(weight_dir, exist_ok=True)
        self.weight_dir = weight_dir
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        for group in self.actor.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_actor
        for group in self.critic.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_critic
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar
        )

    def train(self, gradient_steps: int, batch_size: int = 64, n_weight_logs: int = 20 ) -> None:

        n_total_updates = self._total_timesteps / self.train_freq.frequency * gradient_steps + self.n_priming_steps
        weights_log_interval = n_total_updates // n_weight_logs

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self.n_priming_steps:
                self.on_priming_start()
            for priming_step in range(max(self.n_priming_steps, 1)):

                # Action by the current actor for the sampled state
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.item())

                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                # Compute critic loss
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                assert isinstance(critic_loss, th.Tensor)  # for type checker
                critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Min over all critic networks
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                self._n_updates += 1

                # Update target networks
                if self._n_updates % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                if self._n_updates % weights_log_interval == 0:
                    self.log_weights()
                if self.reset_interval and self._n_updates % self.reset_interval == 0 and self.n_priming_steps == 0:
                    self.reset_model_layers()

            if self.n_priming_steps:
                self.on_priming_end()
                self.n_priming_steps = 0

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def reset_model_layers(self):
        with th.no_grad():
            if self.reset_critic_layers is not None:
                for idx, layer in enumerate([*self.critic.qf0, *self.critic.qf1, *self.critic_target.qf0, *self.critic_target.qf1]):
                    if hasattr(layer, "weight") and (idx in self.reset_critic_layers or "all" in self.reset_critic_layers):
                        layer.reset_parameters()
            if self.reset_actor_layers is not None:
                for idx, layer in enumerate([*self.actor.latent_pi, self.actor.mu, self.actor.log_std]):
                    if hasattr(layer, "weight") and (idx in self.reset_actor_layers or "all" in self.reset_actor_layers):
                        layer.reset_parameters()

    def on_priming_end(self):
        self.reset_model_layers()
        for group in self.critic.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_critic
        for group in self.actor.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_actor

    def on_priming_start(self):
        for group in self.actor.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_actor
        for group in self.critic.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay_critic

    def log_weights(self):
        if self.weight_dir is None:
            return
        for layer_name, layer_weights in itertools.chain(self.actor.state_dict().items(), self.critic.state_dict().items()):
            if "weight" not in layer_name:
                continue
            weights_1d = layer_weights.view(-1)
            weights_1d.time = self.num_timesteps
            weights_1d.n_updates = self._n_updates
            os.makedirs(os.path.join(self.weight_dir, layer_name.replace(".", "_")), exist_ok=True)
            th.save(weights_1d, os.path.join(self.weight_dir, layer_name.replace(".", "_"), str(self._n_updates)))
        pass