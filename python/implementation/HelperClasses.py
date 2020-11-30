import time
from typing import Dict, Any, List, Sequence
import seaborn as sns
from matplotlib import pyplot as plot
import pandas as pd
import numpy as np
import tensorflow as tf

class DataLogger(object):
    """
    Data logging class for debugging and monitoring of training results of RL algorithms.
    Seed: random seed
    episode_variables: "ep": episode number
                        "timestep": timestep of episode
                        "states": veh state variables
                        "actions": actions taken at each state array
                        "rewards": rewards received
    model_variables: "model_parameters": model property parameters
    training_variables: "ep": episode number
                        "timestep": timestep of episode
                        "training_param": training parameters
                        "returns": returns during training
                        "losses": loss functions and values
                        "grad": gradient magnitudes
                        "error": error function of the value function
    execution_timers: "plotting": average time taken for plotting
                      "training": average time taken for training
    """

    def __init__(self,
                 seed: int,
                 episode_variables: Dict = None,
                 model_variables: Dict = None,
                 training_variables: Dict = None,
                 execution_timers: Dict = None):
        self.seed = seed
        self.episode_variables = episode_variables
        self.model_variables = model_variables
        self.training_variables = training_variables
        self.execution_timers = execution_timers




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
        return sum(self.all_timers)/len(self.all_timers)

    def getTimeTaken(self):
        output = "Mean time taken for " + self.timer_name + " = %0.4f" % self.getMeanTime()
        return output

