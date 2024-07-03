import shutil
from copy import deepcopy
from dataclasses import dataclass

import torch
import os

import yaml
from tensorboard import program

from my_util.tracker import Tracker


@dataclass
class Config:

    experiment_tag: str
    environment: str
    n_runs: int
    eval_freq: int
    algorithm: str
    color: str
    plot_titles: list
    is_atari: bool
    n_eval_episodes: int
    learn_args: dict
    agent_args: dict
    time_limit: int = None
    buffer_save_path: str = None
    buffer_load_path: str = None
    model_load_path: str = None
    model_save_path: str = None

    def __post_init__(self):
        if self.n_runs < 1:
            raise ValueError
        if self.n_eval_episodes < 1:
            raise ValueError
        if self.eval_freq < 1:
            raise ValueError
        if self.time_limit is not None and self.time_limit < 1:
            raise ValueError
        if self.buffer_load_path is not None and not os.path.exists(self.buffer_load_path):
            raise ValueError
        if self.model_load_path is not None and not os.path.exists(self.model_load_path):
            raise ValueError


def merge_dicts(dict1, dict2):
    merged_dict = deepcopy(dict1)
    for key, value in dict2.items():
        if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
            merged_dict[key] = merge_dicts(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict


def load_configs(path) -> list[Config]:
    with open(path, 'r') as file:
        config_file = yaml.safe_load(file)

    configs = []
    for tag, experiment in config_file['experiments'].items():
        combined_config = merge_dicts(config_file['constant'], experiment)
        config = Config(tag, **combined_config)
        configs.append(config)

    return configs


def open_tensorboard(log_dir, port=None):
    tb = program.TensorBoard()
    os.makedirs(log_dir, exist_ok=True)
    if port is not None:
        tb.configure(argv=[None, '--logdir', log_dir, '--port', port])
    else:
        tb.configure(argv=[None, '--logdir', log_dir])
    port = tb.launch()
    print(f"Tensorflow listening on {port}")


def copy_file(source_file, destination_dir):
    """
    Copies a file from the current location to the desired destination. Creates the destination directory if it does
    not yet exist.

    Parameters
    ----------
    source_file : str
        Current location of the file.
    destination_dir : str
        Destination location of the file.
    """
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    shutil.copy(source_file, destination_path)