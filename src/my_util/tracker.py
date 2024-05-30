import os.path
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

    def add_datapoint(self, timestep, value):
        if self.data.get(timestep) is None:
            self.data[timestep] = Welford()
        self.data[timestep].add(np.array(value))

    @property
    def times(self):
        return np.array([time for time in self.data.keys()])

    @property
    def means(self):
        return np.array([welford.mean for welford in self.data.values()])

    @property
    def stds(self):
        return np.array([np.sqrt(welford.var_p) for welford in self.data.values()])

    @classmethod
    def plot(cls, plot_title: str, xlabel: str = "Step", ylabel: str = "Episode Return"):
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


class ReturnsTracker:
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
        kwargs :
            arguments required for the metric's add_datapoint function
        """
        self.returns[agent_id].add_datapoint(**kwargs)

    def plot(self, plot_dir: str):
        """
        Plots all metrics and saves the resulting figures

        Parameters
        ----------
        plot_dir :
            Directory where the plots are saved
        """
        os.makedirs(plot_dir, exist_ok=True)
        for plot_title, agents in self.plots.items():
            for agent_id, return_history in agents.items():
                Curve.plot(plot_title=plot_title)
                return_history.add_to_plot()
            plt.legend()
            plt.savefig(os.path.join(plot_dir, plot_title))
            plt.close()

    def save(self, path: str):
        """
        Saves the tracked data for each agent separately.

        Parameters
        ----------
        path :
            Directory where the tracked data is saved
        """
        os.makedirs(path, exist_ok=True)
        for agent_id, return_history in self.returns.items():
            torch.save({agent_id : return_history}, os.path.join(path, agent_id))

    def load(self, config):
        """
        Restores tracked data saved using `save_metrics`.

        Parameters
        ----------
        config :
        """
        agent_id, return_history = tuple(torch.load(config.load_path).items())[0]
        return_history.plot_titles = config.plot_titles
        return_history.label = config.experiment_tag
        return_history.color = config.color
        self.returns.update({config.experiment_tag : return_history})
        for plot in config.plot_titles:
            if self.plots.get(plot) is None:
                self.plots[plot] = {}
            self.plots[plot][agent_id] = self.returns[agent_id]
