from io import StringIO
from csv import writer
from typing import TypeVar
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from torch import argmax, nn
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import polyak_update
from torch.optim import AdamW


class MyDQN(DQN):

    def __init__(
            self,
            double_dqn: bool = True,
            n_priming_steps: int = 0,
            reset_layers: list = None,
            reset_interval: int = None,
            priming_weight_decay: float = 0,
            weight_decay: float = 0,
            n_weight_logs: int = 20,
            n_weight_bins: int = 30,
            *args,
            **kwargs
    ) -> None:
        """

        Parameters
        ----------
        double_dqn : bool
            whether to use DDQN or DQN
        n_priming_steps : int
            Number of policy updates to perform on the first sampled batch
        reset_layers : list
            a list of indices that specify which policy network layers are reset
        reset_target_layers : list
            a list of indices that specify which target network layers are reset
        reset_interval : int
            number of update steps after which network layers are reset
        priming_weight_decay : float
            weight decay regularization factor applied during the priming phase
        weight_decay : float
            weight decay regularization factor applied during the learning phase
        args :
            Arguments passed to the base class constructor
        kwargs :
            Keyword Arguments passed to the base class constructor
        """
        kwargs["policy_kwargs"].update({"optimizer_class" : AdamW})
        super().__init__(
            *args,
            **kwargs
        )
        self.double_dqn = double_dqn
        self.n_priming_steps = n_priming_steps
        self.reset_layers = [] if reset_layers is None else reset_layers
        self.reset_interval = reset_interval
        self.priming_weight_decay = priming_weight_decay
        self.weight_decay = weight_decay
        for group in self.policy.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay

        self.n_weight_logs = n_weight_logs,
        self.n_weight_bins = n_weight_bins
        self.weight_histograms = None

    def _on_step(self) -> None:
        """
        This function is changed from the original implementation of stable-baselines3. It now no longer updates the
        target networks, this is instead periodically performed in MyDQN.train. The purpose of this change was to ensure
        target updates could occur during the heavy priming phase.

        Update the exploration rate.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        for group in self.policy.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay
        self.weight_histograms = {}

        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar
        )

    def train(self, gradient_steps: int, batch_size: int = 100, n_weight_logs: int = 20 ) -> None:

        n_total_updates = self._total_timesteps / self.train_freq.frequency * gradient_steps + self.n_priming_steps
        weights_log_interval = n_total_updates // n_weight_logs

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self.n_priming_steps:
                self._on_priming_start()
            for priming_step in range(max(self.n_priming_steps, 1)):
                with torch.no_grad():
                    if self.double_dqn:
                        greedy_actions = argmax(self.q_net(replay_data.next_observations), dim=1)
                        # Compute the next Q-values using the target network
                        next_q_values = self.q_net_target(replay_data.next_observations)
                        next_q_values = next_q_values[torch.arange(len(next_q_values)), greedy_actions]
                    next_q_values = next_q_values.reshape(-1, 1)
                    # 1-step TD target
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

                loss = F.mse_loss(current_q_values, target_q_values)
                losses.append(loss.item())
                # Optimize the policy
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Increase update counter
                self._n_updates += 1

                if self._n_updates % self.target_update_interval == 0:
                    polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                if self._n_updates % weights_log_interval == 0:
                    self.log_weights()
                if self.reset_interval and self._n_updates % self.reset_interval == 0 and self.n_priming_steps == 0:
                    self.reset_model_layers()
            self._on_priming_end()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def reset_model_layers(self):
        """
        Resets the layers with the specified indices in the policy model and target model.
        """
        with torch.no_grad():
            if self.reset_layers is not None:
                for idx, layer in enumerate(self.policy.q_net.q_net):
                    if hasattr(layer, "weight") and (idx in self.reset_layers or "all" in self.reset_layers):
                        layer.reset_parameters()

    def _on_priming_start(self):
        for group in self.policy.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay

    def _on_priming_end(self):
        if self.n_priming_steps == 0:
            return
        self.n_priming_steps = 0
        self.reset_model_layers()
        for group in self.policy.optimizer.param_groups:
            group['weight_decay'] = self.weight_decay

    def log_weights(self):
        for layer_name, layer_weights in self.q_net.state_dict().items():
            if "weight" not in layer_name:
                continue
            counts, bins = np.histogram(layer_weights.cpu(), bins=self.n_weight_bins)
            row = [self.num_timesteps, self._n_updates, *bins, *counts]
            if self.weight_histograms.get(layer_name) is None:
                bin_keys = [f"bin_{idx}" for idx in range(self.n_weight_bins + 1)]
                count_keys = [f"count_{idx}" for idx in range(self.n_weight_bins)]
                self.weight_histograms[layer_name] = pd.DataFrame([row], columns=["timestep", "n_updates", *bin_keys, *count_keys])
            else:
                self.weight_histograms[layer_name] = pd.concat(
                    [pd.DataFrame([row], columns=self.weight_histograms[layer_name].columns), self.weight_histograms[layer_name]],
                    ignore_index=True
                )