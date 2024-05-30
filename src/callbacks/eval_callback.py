import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):

    """
    Callback that evaluates the policy at a set update step interval
    """

    def __init__(self, eval_freq, n_eval_episodes) -> None:
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards = []
        self.evaluations = {
            "time": [],
            "mean": [],
            "std": []
        }

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.locals['self'].policy.set_training_mode(False)
        if self.locals['self'].num_timesteps % self.eval_freq == 0:
            rewards, episode_lengths = evaluate_policy(
                self.locals['self'],
                self.locals['self'].get_env(),
                self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            self.evaluations['time'].append(self.locals['self']._n_updates)
            self.evaluations['mean'].append(np.mean(rewards))
            self.evaluations['std'].append(np.std(rewards))
            self.locals['self'].logger.record("Greedy undiscounted returns", np.mean(rewards))
        return