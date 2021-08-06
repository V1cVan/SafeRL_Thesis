import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # specify which GPU(s) to be used (1)
import pickle

from hwsim import Simulation, BasicPolicy, StepPolicy, SwayPolicy, IMPolicy, KBModel, TrackPolicy, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot

import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Agents import *
from Policies import *
from HelperClasses import *
from Models import *
import logging
from matplotlib import pyplot as plt
import time
import datetime
import multiprocessing as mp
import sys

tf.config.experimental.set_visible_devices([], "GPU")
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Main(object):

    def __init__(self, n_vehicles, policy, model_param, training_param, tb_logger):
        # ...
        scenario = sim_type(policy=policy, n_vehicles=n_vehicles)
        self.sim = Simulation(scenario)
        self.policy = policy
        # Create object for data logging and visualisation
        self.data_logger = DataLogger(model_param, training_param)
        self.plotTimer = Timer("Plotting")

        self.episodeTimer = Timer("Episode")
        self.custom_action_timer = Timer("Custom_action_timer")
        self.training_timer = Timer("Training_timer")
        self.training_param = training_param
        self.model_param = model_param
        self.tb_logger = tb_logger

    def create_plot(self):
        shape = (4, 2)
        groups = [([0, 2], 0)]
        FANCY_CARS = True
        vehicle_type = "car" if FANCY_CARS else "cuboid3D"
        # self.p = Plotter(self.sim, "Multi car simulation", mode=PLOT_MODE, shape=shape, groups=groups, off_screen=OFF_SCREEN)
        self.p = Plotter(self.sim, "Multi car simulation", mode=Plotter.Mode.LIVE, shape=shape, groups=groups,
                         V=0, state=Plotter.State.PLAY)
        self.p.subplot(0, 0)
        self.p.add_text("Detail view")
        DetailPlot(self.p, show_ids=True)
        self.p.add_overlay()
        SimulationPlot(self.p, vehicle_type=None, show_marker=True)
        self.p.subplot(0, 1)
        self.p.add_text("Front view")
        BirdsEyePlot(self.p, vehicle_type=vehicle_type, view=BirdsEyePlot.View.FRONT)
        self.p.subplot(1, 1)
        self.p.add_text("Rear view")
        BirdsEyePlot(self.p, vehicle_type=vehicle_type, view=BirdsEyePlot.View.REAR)
        self.p.subplot(2, 1)
        self.p.add_text("Actions")
        ActionsPlot(self.p, actions="long")
        self.p.subplot(3, 1)
        ActionsPlot(self.p, actions="lat")
        self.p.subplot(3, 0)
        self.p.add_text("Rewards")
        # TimeChartPlot(self.p, lines=self.policy.agent.buffer.rewards)
        self.p.plot()  # Initial plot

    def train_policy(self):
        running_reward = 0
        episode_count = 1
        plot_freq = self.training_param["plot_freq"]
        show_plots = self.training_param["show_train_plots"]
        max_timesteps = self.training_param["max_timesteps"]
        max_episodes = self.training_param["max_episodes"]
        trained = False
        # episode_list = []
        # running_reward_list = []
        # loss_list = []
        # plot_items = self.data_logger.init_training_plot()
        train_counter = 1
        model_update_counter = 1
        # Run until all episodes are completed (reward reached).
        while episode_count <= max_episodes:
            self.policy.agent.episode = episode_count
            episode_reward_list = []
            episode_mean_batch_rewards = []
            episode_losses = []
            episode_td_errors = []
            vehicle_speeds = []

            if episode_count % plot_freq == 0 and show_plots:
                self.policy.agent.prev_epsilon = self.policy.agent.epsilon
                self.simulate(self.training_param["sim_timesteps"])
                self.policy.agent.epsilon = self.policy.agent.prev_epsilon

            # print("Episode number: %0.2f" % episode_count)
            # logging.critical("Episode number: %0.2f" % episode_count)

            self.episodeTimer.startTime()
            # Set simulation environment
            with self.sim:
                # Loop through each timestep in episode.
                # Run the model for one episode to collect training data
                for t in np.arange(1, max_timesteps+1):
                    # logging.critical("Timestep of episode: %0.2f" % self.sim.k)

                    # Perform one simulations step:
                    if not self.sim.stopped:
                        self.custom_action_timer.startTime()
                        self.sim.step()  # Calls AcPolicy.customAction method.
                        custom_action_time = self.custom_action_timer.endTime()

                        # TODO Reformat so that check is done after sim.step()
                        # TODO maybe add penalties for collisions
                        # self.policy.agent.stop_flags = self.sim.stopped or self.sim._collision
                        # if self.policy.agent.stop_flags == True:
                        #     self.policy.agent.buffer.alter_buffer_stop_flag(flag=self.policy.agent.stop_flags)
                        done = self.sim.stopped  # or self.sim._collision
                        if self.policy.agent.is_action_taken:
                            self.policy.agent.add_experience(done)
                            episode_reward_list.append(self.policy.agent.latest_experience[2])
                            curr_veh_speed = self.sim.vehicles[0].s["vel"][0]*3.6
                            vehicle_speeds.append(curr_veh_speed)

                    if t % self.training_param["policy_rate"] == 0:
                        train_counter += 1
                        self.policy.agent.epsilon_decay_count = train_counter
                        self.policy.agent.timestep = np.int(t / self.training_param["policy_rate"])
                        if self.policy.agent.buffer.is_buffer_min_size():
                            model_update_counter += 1
                            if model_update_counter % self.training_param["model_update_rate"] == 0:
                                # TODO add data to episode buffer to get episode rewards while training.
                                self.training_timer.startTime()
                                mean_batch_reward, loss, td_error, grads, clipped_grads = self.policy.agent.train_step()
                                train_time = self.training_timer.endTime()
                                episode_mean_batch_rewards.append(mean_batch_reward)
                                episode_losses.append(loss)
                                episode_td_errors.append(td_error)
                                trained = True

                            if model_update_counter % self.training_param["target_update_rate"] == 0:
                                self.policy.agent.update_target_net()
                                # print("Updated target net.")

            reward = np.sum(episode_reward_list)/len(episode_reward_list)

            time_taken_episode = self.episodeTimer.endTime()

            if trained:
                if 0.05 * reward + (1 - 0.05) * running_reward > running_reward and episode_count % 50 == 0:
                    self.policy.agent.Q_actual_net.save_weights(self.model_param["weights_file_path"])
                    print("Saved network weights.")

                # Running reward smoothing effect
                running_reward = 0.05 * reward + (1 - 0.05) * running_reward


            if episode_count % 25 == 0 and trained:
                epsilon = self.policy.agent.calc_epsilon()
                if self.training_param["use_per"]:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Beta = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon,
                                                         self.policy.agent.buffer.beta, time_taken_episode)
                else:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon, time_taken_episode)

                print(print_output)
                # logging.critical(print_output)
                # loss_list.append(loss)
                # episode_list.append(episode_count)
                # running_reward_list.append(running_reward)
                # training_var = (episode_list, running_reward_list, loss_list)

            # Save episode training variables to tensorboard
            # TODO Variable to add logging of extended variables if needed
            # self.tb_logger.save_histogram("Episode mean batch rewards", x=episode_count, y=episode_mean_batch_rewards)
            # self.tb_logger.save_histogram("Episode losses", x=episode_count, y=episode_losses)
            # self.tb_logger.save_histogram("Episode TD errors", x=episode_count, y=episode_td_errors)
            self.tb_logger.save_variable("Episode mean batch rewards (sum)", x=episode_count, y=np.sum(episode_mean_batch_rewards))
            self.tb_logger.save_variable("Episode losses (sum)", x=episode_count, y=np.sum(episode_losses))
            self.tb_logger.save_variable("Episode TD errors (sum)", x=episode_count, y=np.sum(episode_td_errors))
            # self.tb_logger.save_variable("Total episode reward (sum)", x=episode_count, y=np.sum(episode_reward_list))
            self.tb_logger.save_variable("Mean episode reward", x=episode_count, y=np.sum(episode_reward_list)/len(episode_reward_list))
            # self.tb_logger.save_variable("Running reward", x=episode_count, y=running_reward)
            self.tb_logger.save_variable("Total time taken for episode", x=episode_count, y=time_taken_episode)
            # self.tb_logger.save_variable("Total time taken for custom action", x=episode_count, y=custom_action_time)
            # self.tb_logger.save_variable("Total time taken for training iteration", x=episode_count, y=train_time)
            self.tb_logger.save_variable("Mean vehicle speed for episode", x=episode_count, y=np.mean(vehicle_speeds))
            if self.training_param["use_per"]:
                self.tb_logger.save_variable("Beta increment", x=episode_count, y=self.policy.agent.buffer.beta)
            # TODO time taken for inferenece and time taken for training step

            # Save model weights and biases and gradients of backprop.
            # TODO fix deepset model so that we can save layer names
            # self.tb_logger.save_weights_gradients(episode=episode_count,
            #                                  model=self.policy.agent.Q_actual_net,
            #                                  grads=grads,
            #                                  clipped_grads=clipped_grads)



                # self.data_logger.save_xls("./models/training_variables.xls")
            if running_reward >= self.training_param["final_return"] \
                    or episode_count == self.training_param["max_episodes"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                # logging.critical(print_output)
                self.policy.agent.Q_actual_net.save_weights(self.model_param["weights_file_path"])
                # pic.dump(training_var, open("./models/train_output", "wb"))
                # self.data_logger.plot_training_data(plot_items)
                # self.data_logger.save_training_data("./models/training_variables.p")
                break

            episode_count += 1

    def simulate(self, simulation_timesteps, simulation_episodes):
        self.policy.agent.training = False
        self.policy.agent.evaluation = True

        episode = 1
        while episode < simulation_episodes:
            steps = 0
            print_counter = 0
            episode_reward_list = []
            vehicle_speeds = []
            with self.sim:
                # self.create_plot()
                while not self.sim.stopped and steps<simulation_timesteps: #and not self.p.closed:
                    self.sim.step()
                    # if print_counter % 10 == 0:
                    #     print(self.policy.agent.epsilon)
                    #     if self.policy.agent.latest_action == 0:
                    #         print("0: Slowing down.")
                    #     elif self.policy.agent.latest_action == 1:
                    #         print("1: Constant speed.")
                    #     elif self.policy.agent.latest_action == 2:
                    #         print("2: Speeding up.")
                    #     elif self.policy.agent.latest_action == 3:
                    #         print("3: Turning left.")
                    #     elif self.policy.agent.latest_action == 4:
                    #         print("4: Turning right.")
                    episode_reward_list.append(self.policy.agent.latest_reward)
                    curr_veh_speed = self.sim.vehicles[0].s["vel"][0] * 3.6
                    vehicle_speeds.append(curr_veh_speed)
                    steps += 1
                    print_counter += 1
                    # self.p.plot()
                # self.p.close()
            episode += 1
            self.tb_logger.save_variable("Mean episode reward", x=episode,
                                         y=np.sum(episode_reward_list) / len(episode_reward_list))
            self.tb_logger.save_variable("Mean vehicle speed for episode", x=episode, y=np.mean(vehicle_speeds))
        self.policy.agent.evaluation = False
        self.policy.agent.training = True

def sim_type(policy, n_vehicles):
    # Randomised highway
    n_slow_veh = n_vehicles["slow"]
    n_normal_veh = n_vehicles["medium"]
    n_fast_veh = n_vehicles["fast"]
    sim_config = {
        "name": "Dense_Highway_Circuit",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": policy, "D_MAX": 160},
            # {"amount": 2, "model": KBModel(), "policy": StepPolicy(10, [0.1, 0.5])},
            # {"amount": 1, "model": KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            # {"amount": 8, "model": KBModel(), "policy": IMPolicy()},
            # {"model": KBModel(), "policy": FixedLanePolicy(18), "R": 0, "l": 3.6, "s": random.random()*200, "v": 18},
            {"amount": n_slow_veh, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": n_normal_veh, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": n_fast_veh, "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    }
    return sim_config

def start_run(arg0, arg1, arg2, arg3):
    # Start training loop using given arguments here..
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")
    config.scenarios_path = str(SC_PATH)
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm")

    if arg1 == "policy_rate":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = arg3
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "batch_size":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = arg3
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "target_update_rate":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = arg3
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "learning_rate":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = arg3
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "loss_function":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = arg3
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "activation":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = arg3
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "model_size":
        N_UNITS = arg3
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
    elif arg1 == "clip_grad":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = arg3
        STANDARDISE_RETURNS = True
    elif arg1 == "standardise_return":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = arg3
    elif arg1 == "gamma":
        N_UNITS = (32, 16, 8)
        ACT_FUNC = tf.nn.relu
        POLICY_ACTION_RATE = 10
        BATCH_SIZE = 32
        TARGET_UPDATE_RATE = 10e4
        LEARN_RATE = 0.0001
        LOSS_FUNC = tf.losses.MeanSquaredError()
        CLIP_GRADIENTS = True
        STANDARDISE_RETURNS = True
        GAMMA = arg3

    """RUN PARAMETERS:"""
    SEED = arg2
    RUN_TYPE = "train"  # "train"
    RUN_NAME = "DDQN_ER_tuning"
    if "MeanSquaredError" in str(arg3):
        RUN_INFO = arg1 + "-" + "MeanSquaredError"
    elif "Huber" in str(arg3):
        RUN_INFO = arg1 + "-" + "Huber"
    elif "relu" in str(arg3):
        RUN_INFO = arg1 + "-" + "Relu"
    elif "swish" in str(arg3):
        RUN_INFO = arg1 + "-" + "Swish"
    elif "tanh" in str(arg3):
        RUN_INFO = arg1 + "-" + "Tanh"
    else:
        RUN_INFO = arg1 + "-" + str(arg3)

    n_vehicles = arg0["slow"]+arg0["medium"]+arg0["fast"]

    if RUN_TYPE == "train":
        SAVE_DIRECTORY = "logfiles/" + RUN_NAME + "/" + RUN_TYPE + "/Seed" + str(SEED) + "-Details=" + str(RUN_INFO)
    else:
        SAVE_DIRECTORY = "logfiles/" + RUN_NAME + "/" + RUN_TYPE + "/Seed" + str(SEED) + "N_veh=" + str(
            n_vehicles) + "-Details=" + str(RUN_INFO)

    run_settings = {
        "run_type": RUN_TYPE,
        "run_name": RUN_NAME,
        "run_info": RUN_INFO,
        "save_directory": SAVE_DIRECTORY,
        "n_vehicles": arg0
    }

    config.seed = SEED
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    """MODEL PARAMETERS:"""
    # N_UNITS = (32, 16, 8)
    N_INPUTS = 55
    N_ACTIONS = 5
    # ACT_FUNC = tf.nn.relu
    N_STACKED_TIMESTEPS = 2  # TODO Check stacked timesteps (shouldnt it be 4? )
    BATCH_NORM = False
    MODEL_FILE_PATH = "logfiles/"+RUN_NAME+"/"+"train"+"/Seed"+str(SEED)+"-Details="+str(RUN_INFO)+"/model_weights"
    if RUN_TYPE == "train":
        TRAINABLE = True
    elif RUN_TYPE == "test":
        TRAINABLE = False
    else:
        print("Run type not correctly specified!")
        sys.exit()

    model_param = {
        "n_units": N_UNITS,
        "n_inputs": N_INPUTS, # TODO change n_units when removing velocity from state measurement!
        "n_actions": N_ACTIONS,
        "activation_function": ACT_FUNC,
        "weights_file_path": MODEL_FILE_PATH,
        "seed": SEED,
        "trainable": TRAINABLE,  # For batch normalisation to freeze layers
        "batch_normalisation": BATCH_NORM,  # TODO maybe add to cnn and lstm maybe + test on deepset + baseline
        "deepset_param":{
            "n_units_phi": (32, 16, 8),
            "n_units_rho": (32, 16, 8)
        },
        "cnn_param": {
            "config": 3,  # 0=1D conv. on vehicle dim.,
            # 1=1D conv. on measurements dim.,
            # 2=2D conv. on vehicle and measurements dimensions,
            # 3=3D conv. on vehicle and measurement dimensions through time
            # Config 0:
            "n_filters_0": 6,  # Dimensionality of output space
            "kernel_size_0": (2,),  # Convolution width
            # Config 1:
            "n_filters_1": 6,  # Dimensionality of output space
            "kernel_size_1": (2,),  # Convolution width
            # Config 2:
            "n_filters_2": 6,  # Dimensionality of output space
            "kernel_size_2": (4, 2),  # Convolution width
            # Config 3:
            "n_filters_3": 6,  # Dimensionality of output space
            "n_timesteps": N_STACKED_TIMESTEPS,
            "kernel_size_3": (N_STACKED_TIMESTEPS, 4, 2)  # Convolution width
        }
    }

    """TRAINING PARAMETERS:"""
    # POLICY_ACTION_RATE = 10  # Number of simulator steps before new control action is taken
    MAX_TIMESTEPS = 5e3  # range: 5e3 - 10e3
    MAX_EPISODES = 1500
    FINAL_RETURN = 1
    SHOW_TRAIN_PLOTS = False
    SAVE_TRAINING = True
    LOG_FREQ = 0
    PLOT_FREQ = 50
    SIM_TIMESTEPS = 5e3
    if RUN_TYPE == 'test':
        SIM_EPISODES = 100
    else:
        SIM_EPISODES = 1
    BUFFER_SIZE = 300000
    # BATCH_SIZE = 32  # range: 32 - 150
    EPSILON_MIN = 1.0  # Exploration
    EPSILON_MAX = 0.1  # Exploitation
    DECAY_RATE = 0.999982  # 0.999992
    MODEL_UPDATE_RATE = 1
    # TARGET_UPDATE_RATE = 10e4
    # LEARN_RATE = 0.0001  # range: 1e-3 - 1e-4
    OPTIMISER = tf.optimizers.Adam(learning_rate=LEARN_RATE)
    # LOSS_FUNC = tf.losses.MeanSquaredError()  # tf.losses.Huber()  # PER loss function is MSE
    # GAMMA = 0.99  # range: 0.95 - 0.99
    # CLIP_GRADIENTS = True
    CLIP_NORM = 2
    # Reward weights = (rew_vel, rew_lat_lane_position, rew_fol_dist, staying_right, collision penalty)
    REWARD_WEIGHTS = np.array([1.0, 0.15, 0.8, 0.4, -5])
    # STANDARDISE_RETURNS = True
    USE_PER = False
    ALPHA = 0.75  # Priority scale: a=0:random, a=1:completely based on priority
    BETA = 0.2  # Prioritisation factor
    BETA_INCREMENT = 0.00004 * MODEL_UPDATE_RATE  # Rate of Beta annealing to 1
    # Model types:
    USE_DUELLING = False
    USE_DEEPSET = False
    USE_CNN = False  # TODO Fix so that CNN type 3 and LSTM work with PER!
    USE_LSTM = False
    FRAME_STACK_TYPE = 0  # 0-Stack last agent action frames, 1=stack simulator frames
    ADD_NOISE = False
    REMOVE_STATE_VELOCITY = False

    training_param = {
        "max_timesteps": MAX_TIMESTEPS,
        "max_episodes": MAX_EPISODES,
        "final_return": FINAL_RETURN,
        "show_train_plots": SHOW_TRAIN_PLOTS,
        "save_training": SAVE_TRAINING,
        "log_freq": LOG_FREQ,
        "plot_freq": PLOT_FREQ,
        "sim_timesteps": SIM_TIMESTEPS,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "epsilon_max": EPSILON_MAX,
        "epsilon_min": EPSILON_MIN,
        "decay_rate": DECAY_RATE,
        "model_update_rate": MODEL_UPDATE_RATE,
        "target_update_rate": TARGET_UPDATE_RATE,
        "policy_rate": POLICY_ACTION_RATE,
        "gamma": GAMMA,
        "clip_gradients": CLIP_GRADIENTS,
        "clip_norm": CLIP_NORM,
        "learning_rate": LEARN_RATE,
        "optimiser": OPTIMISER,
        "loss_func": LOSS_FUNC,
        "seed": SEED,
        "standardise_returns": STANDARDISE_RETURNS,
        "reward_weights": REWARD_WEIGHTS,
        "use_per": USE_PER,
        "alpha": ALPHA,
        "beta": BETA,
        "beta_increment": BETA_INCREMENT,
        "use_duelling": USE_DUELLING,
        "use_deepset": USE_DEEPSET,
        "use_CNN": USE_CNN,
        "use_LSTM": USE_LSTM,
        "frame_stack_type": FRAME_STACK_TYPE,
        "noise_param": {"use_noise": ADD_NOISE, "magnitude": 0.1, "normal": True, "mu": 0.0, "sigma": 0.1,
                        "uniform": True},
        "remove_state_vel": REMOVE_STATE_VELOCITY
    }

    """ INIT SAVING OF ALL TRAINING PARAMETERS AND DATA """
    # Init tensorboard saving of all the training data:
    tb_logger = TbLogger(save_training=SAVE_TRAINING,
                         directory=run_settings["save_directory"],
                         log_freq=LOG_FREQ)
    # Dump model parameters of run:
    pic.dump(model_param, open(run_settings["save_directory"] + "/model_parameters", "wb"))
    # Dump training parameters of run:
    training_param_save = training_param.copy()
    training_param_save.pop("loss_func")
    training_param_save.pop("optimiser")
    pic.dump(training_param_save, open(run_settings["save_directory"] + "/training_parameters", "wb"))

    # TODO Compare DQN DDQN PER and DUELLING ON SAME RANDOM SEED!

    # Initialise model type:
    if USE_DEEPSET and not USE_CNN:
        DQ_net = DeepSetQNetwork(model_param=model_param)
    elif not USE_DEEPSET and USE_CNN:
        DQ_net = CNN(model_param=model_param)
    elif not USE_DEEPSET and not USE_CNN:
        if USE_LSTM:
            DQ_net = LSTM_DRQN(model_param=model_param)
        else:
            if USE_DUELLING:
                DQ_net = DeepSetQNetwork(model_param=model_param)
            else:
                DQ_net = DeepQNetwork(model_param=model_param)
    else:
        print("Error: Cannot use Deepset and CNN methods together!")
        exit()

    # Initialise agent and policy:
    dqn_agent = DqnAgent(network=DQ_net, training_param=training_param, tb_logger=tb_logger)
    dqn_policy = DiscreteSingleActionPolicy(agent=dqn_agent)

    # RewardFunction().plot_reward_functions()

    # TODO check LSTM inputs dimensions to and from buffers and to and from models
    # TODO check deepset dimensions to and from buffers and to and from models
    # TODO check CNN dimensions to and from buffers and to and from models
    # TODO tune deepset, CNN, lstm
    # TODO add CNN mean/max pooling layers
    # TODO check batch normalisation for all the models
    # TODO check to make sure that deepset doesnt need tanh + batch norm ... tanh on other models too
    # Set up main class for running simulations:
    main = Main(n_vehicles=run_settings["n_vehicles"],
                policy=dqn_policy,
                model_param=model_param,
                training_param=training_param,
                tb_logger=tb_logger)

    if run_settings["run_type"] == "train":
        main.policy.agent.evaluation = False
    elif run_settings["run_type"] == "test":
        main.policy.agent.evaluation = True
    else:
        print("Wrong run type inputted")
        sys.exit()

    # Train model:
    if main.policy.agent.evaluation == True:
        main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
        # TODO Tidy up simulation part:
        # Simulate model:
        main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
        main.policy.agent.evaluation = True
        main.simulate(simulation_timesteps=SIM_TIMESTEPS, simulation_episodes=SIM_EPISODES)
    else:
        main.train_policy()

    print(f"Arg0:{arg0}; Arg1:{arg1}; Arg2: {arg2}; Arg3: {arg3}")

if __name__=="__main__":

    PROCS = 32  # Number of cores to use
    mp.set_start_method("spawn")  # Make sure different workers have different seeds if applicable
    P = mp.cpu_count()  # Number of available cores
    procs = max(min(PROCS, P), 1)  # Clip number of procs to [1;P]


    def param_gen():
        # This function yields all the parameter combinations to try
        # For train parameter sweep:
        arg0 = {"slow": 10, "medium": 20, "fast": 5}
        for arg1 in ("policy_rate", "batch_size", "target_update_rate", "learning_rate", "loss_function",
                     "activation", "model_size", "clip_grad", "standardise_return", "gamma"):
            for arg2 in (100, 300, 500):
                if arg1 == "policy_rate":
                    for arg3 in (6, 8, 10):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "batch_size":
                    for arg3 in (32,50,100):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "target_update_rate":
                    for arg3 in (10**3,10**4,10**5):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "learning_rate":
                    for arg3 in (0.001, 0.0005, 0.0001):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "loss_function":
                    for arg3 in (tf.losses.MeanSquaredError(), tf.losses.Huber()):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "activation":
                    for arg3 in (tf.nn.relu, tf.nn.swish, tf.nn.tanh):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "model_size":
                    for arg3 in ((32,32), (64,64), (128,128), (256,256), (32,16,8), (64,32,16), (128,64,32)):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "clip_grad":
                    for arg3 in (True,False):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "standardise_return":
                    for arg3 in (True, False):
                        yield arg0, arg1, arg2, arg3
                elif arg1 == "gamma":
                    for arg3 in (0.9, 0.95, 0.99):
                        yield arg0, arg1, arg2, arg3
                else:
                    sys.exit()

        # For simulation test parameter sweep: # TODO check that simulations work and that the correct values are logged
        # for arg0 in ({"slow": 5, "medium": 15, "fast": 3},
        #              {"slow": 10, "medium": 20, "fast": 5},
        #              {"slow": 15, "medium": 25, "fast": 10},
        #              {"slow": 20, "medium": 30, "fast": 15}):
        #     for arg1 in ("policy_rate", "batch_size", "target_update_rate", "learning_rate", "loss_function",
        #                  "activation", "model_size", "clip_grad", "standardise_return"):
        #         for arg2 in (100, 300, 500):
        #             if arg1 == "policy_rate":
        #                 for arg3 in (6, 8, 10):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "batch_size":
        #                 for arg3 in (32, 50, 100):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "target_update_rate":
        #                 for arg3 in (10 ** 3, 10 ** 4, 10 ** 5):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "learning_rate":
        #                 for arg3 in (0.001, 0.0005, 0.0001):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "loss_function":
        #                 for arg3 in (tf.losses.MeanSquaredError(), tf.losses.Huber()):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "activation":
        #                 for arg3 in (tf.nn.relu, tf.nn.swish, tf.nn.tanh):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "model_size":
        #                 for arg3 in (
        #                 (32, 32), (64, 64), (128, 128), (256, 256), (32, 16, 8), (64, 32, 16), (128, 64, 32)):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "clip_grad":
        #                 for arg3 in (True, False):
        #                     yield arg0, arg1, arg2, arg3
        #             elif arg1 == "standardise_return":
        #                 for arg3 in (True, False):
        #                     yield arg0, arg1, arg2, arg3
        #             else:
        #                 sys.exit()

    if procs > 1:
        # Schedule all training runs in a parallel loop over multiple cores:
        with mp.Pool(processes=procs) as pool:
            pool.starmap(start_run, param_gen())
            pool.close()
            pool.join()
    else:
        # Schedule on a single core:
        for args in param_gen():
            start_run(*args)

    print("EOF")