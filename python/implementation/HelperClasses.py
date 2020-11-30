import time
from typing import Dict, Any, List, Sequence
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


class DataLogger(object):
    """
    Data logging class for debugging and monitoring of training results of RL algorithms.
    Seed: random seed
    global_variables: "seed": seed for random variables reproducability
    episode_variables: "ep": episode number
                        "timesteps": timestep of episode
                        "states": veh state variables
                        "sim_actions": actions taken by simulator for each state array
                        "pol_actions": actions taken by policy for each state array
                        "rewards": rewards received
    model_variables: "parameters": model property parameters
                     "summary": Keras summary print of model
                     "weights": Weights of trained model
    training_variables: "parameters": training parameters
                        "returns": returns during training
                        "losses": loss functions and values
                        "grad": gradient magnitudes
                        "error": error function of the value function
    execution_timers: "plotting": average time taken for plotting
                      "training": average time taken for training
    """

    def __init__(self, seed: int, model_parameters: Dict, training_parameters: Dict):
        self.global_variables = {
            "seed": seed
        }
        self.episode_variables = {
            "ep": [],
            "timestamps": [],
            "states": [],
            "sim_actions": [],
            "pol_actions": [],
            "rewards": []
        }
        self.model_variables = {
            "parameters": model_parameters,
            "summary": "",
            "weights": []
        }
        self.training_variables = {
            "parameters": training_parameters,
            "returns": [],
            "losses": [],
            "grad": [],
            "error": []
        }
        self.execution_timers = {
            "plotting": [],
            "training": []
        }

    def set_complete_episode(self, episode: int, timestamps: List, states: List,
                             sim_actions: List, pol_actions: List, rewards: List):
        """ Sets the training data for a complete episode. """
        self.training_variables["episode"].append(episode)
        self.training_variables["timestamps"] = timestamps
        self.training_variables["states"] = states
        self.training_variables["sim_actions"] = sim_actions
        self.training_variables["pol_actions"] = pol_actions
        self.training_variables["rewards"] = rewards

    def display_model_overview(self, keras_summary):
        """ Displays keras model summary, model parameters, and weights for each layer. """
        print()

    def plot_training_rewards(self):
        """ Plots the mean reward per episode during training. """
        plt.figure()
        # Calculate mean reward per episode
        mean_reward = [np.mean(x) for x in self.training_variables["rewards"]]
        sns.tsplot(time=self.training_variables["episode"], data=mean_reward,
                   linestyle='-', linewidth=2)
        plt.grid(True)
        plt.title("Total reward per episode.")
        plt.xlabel("Episode number")
        plt.ylabel("Reward")


class Timer(object):
    def __init__(self, timer_name):
        self.timer_name = timer_name
        self.tic = None
        self.toc = None
        self.all_timers = []

    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.perf_counter()
        self.all_timers.append(self.toc - self.tic)

    def startTime(self):
        self.tic = time.perf_counter()

    def endTime(self):
        self.toc = time.perf_counter()
        self.all_timers.append(self.toc - self.tic)

    def getMeanTime(self):
        return sum(self.all_timers) / len(self.all_timers)

    def getTimeTaken(self):
        output = "Mean time taken for " + self.timer_name + " = %0.4f" % self.getMeanTime()
        return output
