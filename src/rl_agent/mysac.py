from torch.nn import functional as F
import torch as th
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name

from my_util.tracker import ReturnsTracker


class MySAC(SAC):

    def __init__(
            self,
            n_priming_steps: int = 1,
            reset_actor_layers=None,
            reset_critic_layers=None,
            reset_critic_target_layers=None,
            reset_interval=None,
            priming_weight_decay: float = 0,
            weight_decay: float = 0,
            plot_distributions: bool = False,
            tracker: ReturnsTracker = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            *args,
            **kwargs
        )
        self.n_priming_steps = n_priming_steps
        self.reset_actor_layers = reset_actor_layers
        self.reset_critic_target_layers = reset_critic_target_layers
        self.reset_critic_layers = reset_critic_layers
        self.reset_interval = reset_interval

        # Regularization
        self.priming_weight_decay = priming_weight_decay
        self.weight_decay = weight_decay

        self.histograms = {}
        self.log_interval = None
        self.plot_distributions = plot_distributions
        self.tracker = tracker

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        self.log_interval = max(
            ((total_timesteps / self.train_freq.frequency) * self.gradient_steps + self.n_priming_steps) // 20, 1)
        self.on_learning_start()
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
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

            for priming_step in range(max(self.n_priming_steps, 1)):
                if (self._n_updates + priming_step) % self.log_interval == 0:
                    self.log_weight_distributions(self._n_updates + priming_step - self.n_priming_steps)

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    self.actor.reset_noise()

                # Action by the current actor for the sampled state
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                # Optimize entropy coefficient, also called
                # entropy temperature or alpha in the paper
                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

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

                # Update target networks
                if self._n_updates % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self.on_priming_end()
            self.n_priming_steps = 0
            self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def reset_model_layers(self):
        pass

    def _should_reset(self):
        return self._n_updates == 0 or (
                    self.reset_interval is not None and self.num_timesteps % self.reset_interval == 0)

    def on_priming_end(self):
        if self._should_reset():
            self.reset_model_layers()

    def on_learning_start(self):
        pass

    def log_weight_distributions(self, timestep):
        if not self.plot_distributions:
            return
        for key, layer in self.policy.state_dict().items():
            if "weight" not in key:
                continue
            if self.plot_distributions:
                self.tracker.add_metric(key, metric_type="ridgeline", plots=[key], label=key)
                counts, bins = np.histogram(layer.cpu(), bins=30)
                self.tracker.add_datapoint(key, counts=counts, bin_edges=bins, step=timestep)