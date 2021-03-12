import time
from typing import Dict, List
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle as pic
from collections import deque
import random

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
    global_variables: "seed": seed for random variables reproducibility
    episodes - list of all the data of an episode
    episode_variables:
                        "ep": episode number
                        "timesteps": timestep of episode
                        "states": veh state variables
                        "network_outputs": raw network outputs
                        "sim_actions": actions taken by simulator for each state array
                        "pol_actions": actions taken by policy for each state array
                        "rewards": rewards received
    model_variables:
                        "parameters": model property parameters
                        "summary": Keras summary print of model
                        "weights": Weights of trained model
    training_variables:
                        "parameters": training parameters
                        "returns": returns during training
                        "losses": loss functions and values
                        "grad": gradient magnitudes
                        "error": error function of the value function
    execution_timers:
                        "plotting": average time taken for plotting
                        "training": average time taken for training
    """

    def __init__(self, model_parameters: Dict, training_parameters: Dict):
        self.global_parameters = {
            "seed": training_parameters["seed"],
            "training_parameters": training_parameters,
            "model_parameters": model_parameters,
            "model_summary": ""
        }

        # TODO Change variables to dataframes from start of usage in Datalogger class!

        self.episodes = []
        # self.episode_variables = {
        #     "ep": [],
        #     "timesteps": [],
        #     "states": [],
        #     "network_outputs": [],
        #     "sim_actions": [],
        #     "pol_actions": [],
        #     "ep_rewards": []
        # }
        # self.model_variables = {
        #     "weights": []
        # }
        # self.training_variables = {
        #     "returns": [],
        #     "losses": [],
        #     "grad": [],
        #     "error": []
        # }
        self.execution_timers = {
            "plotting": [],
            "training": []
        }

    def set_complete_episode(self, episode_data):
        """ Sets the training data for a complete episode. """
        self.episodes.append(episode_data)

    def save_xls(self, xls_path):
        """ Not useful """
        timesteps = []
        vel_model = []
        offset_model = []
        vel_simulator = []
        offset_simulator = []
        vel_choice = []
        offset_choice = []
        rewards = []
        critic = []
        advantage = []
        losses = []
        returns = []
        gradients = []
        for ep in self.episodes:
            timesteps.append(ep["timestep"])
            vel_model.append(ep["action"]["vel_model"])
            offset_model.append(ep["action"]["offset_model"])
            vel_simulator.append(ep["action"]["vel_simulator"])
            offset_simulator.append(ep["action"]["offset_simulator"])
            vel_choice.append(ep["action"]["vel_choice"])
            offset_choice.append(ep["action"]["offset_choice"])
            rewards.append(ep["reward"])
            critic.append(ep["critic"])
            advantage.append(ep["advantage"])
            losses.append(ep["losses"])
            gradients.append(ep["gradients"])
        timesteps = pd.DataFrame(timesteps).transpose()
        vel_model = pd.DataFrame(vel_model).transpose()
        offset_model = pd.DataFrame(offset_model).transpose()
        vel_simulator = pd.DataFrame(vel_simulator).transpose()
        offset_simulator = pd.DataFrame(offset_simulator).transpose()
        vel_choice = pd.DataFrame(vel_choice).transpose()
        offset_choice = pd.DataFrame(offset_choice).transpose()
        rewards = pd.DataFrame(rewards).transpose()
        critic = pd.DataFrame(critic).transpose()
        advantage = pd.DataFrame(advantage).transpose()
        losses = pd.DataFrame(losses).transpose()
        returns = pd.DataFrame(returns).transpose()
        gradients = pd.DataFrame(gradients).transpose()
        episodes = [timesteps,
                    vel_model,
                    offset_model,
                    vel_simulator,
                    offset_simulator,
                    vel_choice,
                    offset_choice,
                    rewards,
                    critic,
                    advantage,
                    losses,
                    returns]

        with pd.ExcelWriter(xls_path) as writer:
            for n, df in enumerate(episodes):
                df.to_excel(writer, 'sheet%s' % n)
            writer.save()

    def save_training_data(self, file_dir):
        """"
        Pickles the all the variables during training for all the episodes in a file that can be opened later.
        """
        self.global_parameters["training_parameters"].pop("optimiser")
        training_variables = {
            "parameters": self.global_parameters,
            "episode_variables": self.episodes,
            "timers": self.execution_timers
        }
        pic.dump(training_variables, open(file_dir, "wb"))

    def load_training_data(self, file_dir):
        """"
        Loads past training data.
        """
        training_variables = pic.load(open(file_dir, "rb"))
        return training_variables

    def display_model_overview(self, keras_summary):
        """ Displays keras model summary, model parameters, and weights for each layer. """
        print()

    def init_training_plot(self):
        fig = plt.figure(0,figsize=(18,12))

        rewards_graph = fig.add_subplot(221)
        rewards_graph.set_autoscale_on(True) # enable autoscale
        rewards_graph.autoscale_view(True,True,True)
        r_lines, = rewards_graph.plot([],[],'r.-')
        plt.title("Total rewards per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.pause(0.001)
        advantage_graph = fig.add_subplot(222)
        advantage_graph.set_autoscale_on(True)  # enable autoscale
        advantage_graph.autoscale_view(True, True, True)
        a_lines, = advantage_graph.plot([], [], 'b.-')
        plt.title("Average advantage (Temporal Diff.) per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Advantage")
        plt.grid(True)
        plt.pause(0.001)
        losses_graph = fig.add_subplot(223)
        losses_graph.set_autoscale_on(True)  # enable autoscale
        losses_graph.autoscale_view(True, True, True)
        l_lines, = losses_graph.plot([], [], 'g.-')
        plt.title("Average loss (objective scalar) per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.pause(0.001)
        grad_graph = fig.add_subplot(224)
        grad_graph.set_autoscale_on(True)  # enable autoscale
        grad_graph.autoscale_view(True, True, True)
        g_lines, = grad_graph.plot([], [], '.-')
        plt.title("Average gradient value per set of gradients per episode.")
        plt.xlabel("Episode")
        plt.ylabel("Gradients.")
        plt.grid(True)
        plt.pause(0.001)
        plt.ion()


        lines = [r_lines, a_lines, l_lines, g_lines]
        axes = [rewards_graph, advantage_graph, losses_graph, grad_graph]

        return [fig, axes, lines]

    def plot_training_data(self, plot_items):
        """ Plots the training data."""

        fig, axes, lines = tuple(plot_items)
        rewards_graph, advantage_graph, losses_graph, grad_graph = tuple(axes)
        r_lines, a_lines, l_lines, g_lines = tuple(lines)

        num_episodes = len(self.episodes)
        ep = np.arange(1, num_episodes+1)

        # TODO confidence intervals?

        rewards_sum = []
        advantage_mean = [np.mean(j) for j in self.episodes[0]["advantage"]]
        losses = []
        gradients_mean = []

        for i in ep:
            rewards_sum.append( self.episodes[i-1]["episode_reward"])
            losses.append(self.episodes[i-1]["losses"])
            grad_layers_mean = []
            for j in np.arange(len(self.episodes[i-1]["gradients"][0])):
                grad_layers_mean.append(np.mean(self.episodes[i-1]["gradients"][0][j]))
            gradients_mean.append(np.array(grad_layers_mean))

        fig.canvas.flush_events()

        r_lines.set_data(ep, rewards_sum)
        rewards_graph.relim()  # Recalculate limits
        rewards_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        a_lines.set_data(ep, advantage_mean)
        advantage_graph.relim()  # Recalculate limits
        advantage_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        l_lines.set_data(ep, losses)
        losses_graph.relim()  # Recalculate limits
        losses_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        gradients_mean = np.array(gradients_mean)
        # if gradients_mean.all()!=np.NaN:
        #     g_lines.set_data(ep, np.mean(gradients_mean, axis=1))
        #     grad_graph.relim()  # Recalculate limits
        #     grad_graph.autoscale_view(True, True, True)  # Autoscale
        plt.pause(0.001)
        plt.draw()
        plt.savefig('Training_plots.png')


class EpisodeBuffer(object):
    """
    EpisodeBuffer class (temporarily) saves and handles the experiences per episode during training.
    """

    def __init__(self):
        self.episode = 1
        self.timesteps = []
        self.states = []
        self.actions = {
            "model": [],
            "simulator": [],
            "choice": [],
        }
        self.critic = []
        self.advantage = []
        self.rewards = []
        self.episode_reward = []
        self.loss = []
        self.returns = []
        self.grads = []
        self.model_weights = []
        self.experience = None

    def set_experience(self, timestep, state, action_model,
                       action_sim, action_choice, reward, critic):
        """ Adds the most recent experience to the buffer. """
        self.timesteps.append(timestep)
        self.states.append(state)
        self.actions["model"].append(action_model)
        self.actions["simulator"].append(action_sim)
        self.actions["choice"].append(action_choice)
        self.rewards.append(reward)
        self.critic.append(critic)

    def set_training_variables(self, episode_num, episode_reward, losses, advantage, returns, gradients, model_weights):
        self.episode = episode_num
        self.episode_reward = episode_reward
        self.loss.append(losses)
        self.advantage.append(advantage)
        self.returns.append(returns)
        self.grads.append(gradients)
        self.model_weights.append(model_weights)

    def get_experience(self, timestep: int = None):
        """ Returns the experience at the provided timestep. """
        if timestep is not None:
            index = timestep-1
            actions = {
                "model": self.actions["model"][index],
                "simulator": self.actions["simulator"][index],
                "choice": self.actions["choice"][index],
            }
            output_dict = {
                "episode_num": self.episode,
                "timestep": self.timesteps[index],
                "action": actions,
                "state": self.states[index],
                "reward": self.rewards[index],
                "critic": self.critic[index],
                "advantage": self.advantage[index],
                "losses": None,
                "returns": self.returns[index],
                "gradients": self.grads[index],
                "model_weights": self.model_weights[index]
            }
        else:
            actions = {
                "model": self.actions["model"],
                "simulator": self.actions["simulator"],
                "choice": self.actions["choice"],
            }
            output_dict = {
                "episode_num": self.episode,
                "timestep": self.timesteps,             # 1000
                "action": actions,                      # 1000x...
                "state": self.states,                   # 1000,54
                "reward": self.rewards,                 # 1000
                "episode_reward": self.episode_reward,
                "critic": self.critic,                  # 1000
                "advantage": self.advantage,
                "losses": self.loss,                    # 2
                "returns": self.returns,                # 2,1000
                "gradients": self.grads,                # 2,14
                "model_weights": self.model_weights     # 2,14
            }
        return output_dict

    def set_tf_experience_for_episode_training(self):
        timesteps = tf.convert_to_tensor(self.timesteps)
        states = tf.convert_to_tensor(self.states)
        rewards = tf.convert_to_tensor(self.rewards)
        action_choice = tf.convert_to_tensor(self.actions["choice"], dtype=tf.int32)
        self.experience = timesteps, states, rewards, action_choice

    def alter_reward_at_timestep(self, timestep, reward_change):
        """
        Sets a custom reward at a specific training timestep e.g. collision punishments.
        """
        index = timestep-1
        self.rewards[index] = self.rewards[index] + reward_change

    def clear_experience(self):
        """ Clears all the experiences in the episode. """
        self.timesteps = []
        self.states = []
        self.actions["model"] = []
        self.actions["simulator"] = []
        self.actions["choice"] = []
        self.rewards = []
        self.critic = []
        self.loss = []
        self.returns = []
        self.grads = []
        self.model_weights = []



class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, max_mem_size, batch_size):
        self.max_mem_size = max_mem_size
        self.buffer = deque(maxlen=max_mem_size)
        self.batch_size = batch_size

    def set_experience(self, state, action, reward, next_state, done_flag):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        experience = (state, action, reward, next_state, done_flag)
        self.buffer.append(experience)

    def get_training_samples(self):
        """ Get minibatch for training. """
        mini_batch = random.sample(self.buffer, self.batch_size)
        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch])))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch], dtype=np.float32)))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
        done = tf.cast([each[4] for each in mini_batch], dtype=tf.float32)
        return states, actions, rewards, next_states, done

    def alter_buffer_stop_flag(self, flag):
        state, action, reward, next_state, done_flag = self.buffer[-1]
        done_flag = flag
        self.buffer[-1] = state, action, reward, next_state, done_flag

    def is_buffer_min_size(self):
        return len(self.buffer) >= self.batch_size



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
