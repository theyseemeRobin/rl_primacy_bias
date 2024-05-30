import argparse
import os
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import torch.cuda
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from callbacks.eval_callback import EvalCallback
from my_util.tracker import ReturnsTracker
from my_util.utils import Config, load_configs, open_tensorboard, copy_file
from rl_agent.mydqn import MyDQN
from rl_agent.mysac import MySAC

parser = argparse.ArgumentParser(description="Deep learning with primacy bias")
parser.add_argument("--run-id", type=str, default="DRL_experiments", help="Identifier for the run. Used for naming files/folders.")
parser.add_argument("--config", type=str, default="data/dqn_config.yaml", help="File that contains the agent configurations.")
parser.add_argument("--port", type=str, default="6006", help="value that determines the port to use for tensorboard.")
args = parser.parse_args()


algorithms = {
    "ddqn" : MyDQN,
    "sac" : MySAC,
}


def main(config: Config, tracker: ReturnsTracker, log_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Apply stable baselines' preprocessing steps for atari environments
    if config.is_atari:
        env = make_atari_env(config.environment)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = gym.make(config.environment, render_mode="rgb_array")
    if config.time_limit:
        env = TimeLimit(env, config.time_limit)

    for run in range(config.n_runs):
        tensorboard_dir = None if log_dir is None else os.path.join(log_dir, "tensorboard", config.experiment_tag)
        eval_callback = EvalCallback(config.eval_freq, config.n_eval_episodes)
        agent = algorithms[config.algorithm](
            env=env,
            device=device,
            verbose=False,
            tensorboard_log=tensorboard_dir,
            **config.agent_args
        )
        if config.buffer_load_path:
            agent.load_replay_buffer(config.buffer_load_path)
        if config.model_load_path:
            # remove output layers to allow transfer across different environments
            loaded_model = {key: value for key, value in torch.load(config.model_load_path).items() if
                            'q_net.0' not in key and 'q_net_target.0' not in key}
            new_model = agent.policy.state_dict()
            new_model.update(loaded_model)
            agent.policy.load_state_dict(new_model)

        # Execute the learning process
        agent.learn(
            callback=CallbackList([eval_callback]),
            progress_bar=True,
            tb_log_name=config.experiment_tag,
            **config.learn_args
        )

        # Add the data gathered by the evaluation callback to the tracker
        for time, ep_return in zip(eval_callback.evaluations['time'], eval_callback.evaluations['mean']):
            tracker.add_datapoint(config.experiment_tag, timestep=time, value=ep_return)

        # Save the model
        if config.model_save_path is None:
            save_path = os.path.join(log_dir, "models", config.experiment_tag, f"{config.experiment_tag}_{run}", "model")
        else:
            save_path = config.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            agent.policy.state_dict(),
            save_path
        )
    if config.buffer_save_path:
        agent.save_replay_buffer(config.buffer_save_path)


if __name__ == "__main__":

    base_output_path = os.path.join("data", f"{args.run_id}")
    print(f"Saving results to: {base_output_path}")
    copy_file(args.config, base_output_path)
    open_tensorboard(os.path.join(base_output_path, "tensorboard"), port=args.port)

    configs = load_configs(args.config)

    plots = {}
    experiment_results = {}
    tracker = ReturnsTracker()
    for config in configs:
        if config.load_path is not None:
            tracker.load(config)
        else:
            tracker.add_agent(config.experiment_tag, config.plot_titles, color=config.color, label=config.experiment_tag)
            main(config, tracker, base_output_path)
        tracker.save(base_output_path)
    tracker.plot(os.path.join(base_output_path, "plots"))
