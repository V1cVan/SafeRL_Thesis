import time
from typing import Dict, List
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

"""
    HelperClasses.py module contains:
        DataLogger - for logging, saving and plotting of all RL training data during runtime.
        Buffer - for temporarily handling and data during a RL episode.
        Timer - for timing of python tasks.
"""


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



class Buffer(object):
    """
    Buffer class (temporarily) saves and handles the experiences per episode during training.
    """

    def __init__(self):
        self.timesteps = []
        self.states = []
        self.actions = {
            "vel_model": [],
            "offset_model": [],
            "vel_simulator": [],
            "offset_simulator": [],
            "vel_choice": [],
            "offset_choice": []
        }
        self.critic = []
        self.rewards = []
        self.experience = None

    def add_experience(self, timestep, state, vel_model_action, off_model_action,
                       vel_action_sim, offset_action_sim, vel_choice, off_choice, reward, critic):
        """ Adds the most recent experience to the buffer. """
        self.timesteps.append(timestep)
        self.states.append(state)
        self.actions["vel_model"].append(vel_model_action)
        self.actions["offset_model"].append(off_model_action)
        self.actions["vel_simulator"].append(vel_action_sim)
        self.actions["offset_simulator"].append(offset_action_sim)
        self.actions["vel_choice"].append(vel_choice)
        self.actions["offset_choice"].append(off_choice)
        self.rewards.append(reward)
        self.critic.append(critic)

    def get_experience(self, timestep: int = None):
        """ Returns the experience at the provided timestep. """
        if timestep is not None:
            index = timestep-1
            actions = {
                "vel_model": self.actions["vel_model"][index],
                "offset_model": self.actions["offset_model"][index],
                "vel_simulator": self.actions["vel_simulator"][index],
                "offset_simulator": self.actions["offset_simulator"][index],
                "vel_choice": self.actions["vel_choice"][index],
                "offset_choice": self.actions["offset_choice"][index],
            }
            output_dict = {
                "timestep": self.timesteps[index],
                "action": actions,
                "state": self.states[index],
                "reward": self.rewards[index],
                "critic": self.critic[index]
            }
        else:
            actions = {
                "vel_model": self.actions["vel_model"],
                "offset_model": self.actions["offset_model"],
                "vel_simulator": self.actions["vel_simulator"],
                "offset_simulator": self.actions["offset_simulator"],
                "vel_choice": self.actions["vel_choice"],
                "offset_choice": self.actions["offset_choice"],
            }
            output_dict = {
                "timestep": self.timesteps,
                "action": actions,
                "state": self.states,
                "reward": self.rewards,
                "critic": self.critic
            }
        return output_dict

    def set_tf_experience_for_episode_training(self):
        timesteps = tf.convert_to_tensor(self.timesteps)
        states = tf.convert_to_tensor(self.states)
        rewards = tf.convert_to_tensor(self.rewards)
        action_vel_choice = tf.expand_dims(tf.convert_to_tensor(self.actions["vel_choice"], dtype=tf.int32),1)
        action_off_choice = tf.expand_dims(tf.convert_to_tensor(self.actions["offset_choice"], dtype=tf.int32),1)
        self.experience = timesteps, states, rewards, action_vel_choice, action_off_choice

    def alter_reward_at_timestep(self, timestep, reward_change):
        """
        Sets a custom reward at a specific training timestep e.g. collision punishments.
        """
        index = timestep-1
        self.rewards[index] = self.rewards[index] + reward_change

    def clear_experience(self):
        """ Clears all the experiences in the episode. """
        self.timesteps.clear()
        self.states.clear()
        self.actions["vel_model"].clear()
        self.actions["offset_model"].clear()
        self.actions["vel_simulator"].clear()
        self.actions["offset_simulator"].clear()
        self.actions["vel_choice"].clear()
        self.actions["offset_choice"].clear()
        self.rewards.clear()
        self.critic.clear()




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
