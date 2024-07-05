import os.path

import joypy
import pandas as pd
import torch
from matplotlib import pyplot as plt
from welford import Welford
import numpy as np


class Curve:

    def __init__(self, label: str, color, plot_titles: str):
        self.label = label
        self.color = color
        self.plot_titles = plot_titles
        self.data = {}

    def add_datapoint(self, timestep, update_step, value):
        if self.data.get((timestep, update_step)) is None:
            self.data[(timestep, update_step)] = Welford()
        self.data[(timestep, update_step)].add(np.array(value))

    @property
    def times(self):
        return np.array([time for time, update_step in self.data.keys()])

    @property
    def update_steps(self):
        return np.array([update_step for time, update_step in self.data.keys()])

    @property
    def means(self):
        return np.array([welford.mean for welford in self.data.values()])

    @property
    def stds(self):
        return np.array([np.sqrt(welford.var_p) for welford in self.data.values()])

    @classmethod
    def plot(cls, plot_title: str, xlabel: str = "Update Step", ylabel: str = "Return"):
        plt.rcParams['font.family'] = 'serif'
        if xlabel:
            plt.xlabel(xlabel, fontsize=18)
        if ylabel:
            plt.ylabel(ylabel, fontsize=18)
        if plot_title:
            plt.title(plot_title, fontsize=18)
        plt.grid(True, linestyle='--')
        plt.gca().spines['top'].set_color('#BBBBBB')
        plt.gca().spines['right'].set_color('#BBBBBB')
        plt.gca().spines['left'].set_color('#BBBBBB')
        plt.gca().spines['bottom'].set_color('#BBBBBB')
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.tick_params(axis='both', length=0, labelsize=18, pad=8)
        plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.93)

    def add_to_plot(self):
        plt.plot(self.times, self.means, label=self.label, color=self.color)
        plt.fill_between(self.times, self.means - self.stds, self.means + self.stds, color=self.color,
                         alpha=0.3, linewidth=0)

    def to_csv(self, path):
        csv_data = np.stack((self.times, self.update_steps, self.means, self.stds), axis=1)
        df = pd.DataFrame(csv_data, columns=["time", "update_step", "mean", "std"])
        df.to_csv(path, index=False)


class Tracker:
    """
    Tracks mean and std returns over multiple runs for specified agents.
    """

    def __init__(self):
        self.returns = {}
        self.plots = {}

    def add_agent(self, agent_id: str, plots=None, **metric_kwargs):
        """
        Adds an agent for tracking.

        Parameters
        ----------
        agent_id :
            name of the added agent
        plots :
            title of the plot this metric should be added to
        metric_kwargs :
            arguments for the Curve constructor
        """
        if plots is None:
            plots = ["returns"]
        if self.returns.get(agent_id) is None:
            self.returns[agent_id] = Curve(plot_titles=plots, **metric_kwargs)
        for plot in plots:
            if self.plots.get(plot) is None:
                self.plots[plot] = {}
            self.plots[plot][agent_id] = self.returns[agent_id]

    def add_datapoint(self, agent_id: str = None, **kwargs):
        """
        Adds a datapoint to the specified metric.

        Parameters
        ----------
        agent_id :
            name of the agent whose metric is modified
hist        kwargs :
            arguments required for the metric's add_datapoint function
        """
        self.returns[agent_id].add_datapoint(**kwargs)

    def plot_returns(self, plot_dir: str, weights_dir):
        """
        Plots all metrics and saves the resulting figures

        Parameters
        ----------
        plot_dir :
            Directory where the plots are saved
        """
        os.makedirs(os.path.join(plot_dir, "returns"), exist_ok=True)
        os.makedirs(os.path.join(plot_dir, "distributions"), exist_ok=True)
        for plot_title, agents in self.plots.items():
            for agent_id, return_history in agents.items():
                Curve.plot(plot_title=plot_title)
                return_history.add_to_plot()
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "returns", plot_title.replace(".", "_")))
            plt.close()

    def save(self, path: str):
        """
        Saves the tracked data for each agent separately.

        Parameters
        ----------
        path :
            Directory where the tracked data is saved
        """
        for agent_id, return_history in self.returns.items():
            os.makedirs(os.path.join(path, agent_id, "returns"), exist_ok=True)
            return_history.to_csv(os.path.join(path, agent_id, "returns", agent_id + ".csv"))


def plot_weights(plot_dir, weights_dir):
    for layer_name in os.listdir(weights_dir):
        folder_path = os.path.join(weights_dir, layer_name)
        if os.path.isdir(folder_path):
            data = np.array([])
            steps = np.array([])
            for weight_file in os.listdir(folder_path):
                weights = torch.load(os.path.join(folder_path, weight_file))
                steps = np.concatenate((steps, np.repeat(weights.n_updates, len(weights))))
                data = np.concatenate((data, weights.cpu().numpy()))
            df = pd.DataFrame({"weights": data, "n_updates": steps})
            fig, axes = joypy.joyplot(df, colormap=plt.colormaps.get_cmap("autumn"), by="n_updates")
            for idx, ax in enumerate(axes):
                ax.set_yticklabels([int(float(ax.get_yticklabels()[0].get_text()))])
                if idx % 2 != 0:
                    ax.set_yticklabels("")
            plt.xlabel(f"Model weights", fontsize=16)
            plt.ylabel("Update Step", fontsize=16)
            plt.title(layer_name, fontsize=20)
            os.makedirs(os.path.join(plot_dir), exist_ok=True)
            plt.subplots_adjust(top=0.9, bottom=0.1)
            plt.savefig(os.path.join(plot_dir, str(layer_name).replace(".", "_")))
            plt.close()