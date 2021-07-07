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

# tf.config.experimental.set_visible_devices([0], "GPU")

class Main(object):

    def __init__(self, scenario_num, policy):
        # ...
        scenario = sim_types(scenario_num=scenario_num, policy=policy)
        self.sim = Simulation(scenario)
        self.policy = policy
        # Create object for data logging and visualisation
        self.data_logger = DataLogger(model_param, training_param)
        self.plotTimer = Timer("Plotting")

        self.episodeTimer = Timer("Episode")

    # ...
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
        plot_freq = training_param["plot_freq"]
        show_plots = training_param["show_train_plots"]
        max_timesteps = training_param["max_timesteps"]
        max_episodes = training_param["max_episodes"]
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
                self.simulate(training_param["sim_timesteps"])
                self.policy.agent.epsilon = self.policy.agent.prev_epsilon

            # print("Episode number: %0.2f" % episode_count)
            logging.critical("Episode number: %0.2f" % episode_count)

            self.episodeTimer.startTime()
            # Set simulation environment
            with self.sim:
                # Loop through each timestep in episode.

                # Run the model for one episode to collect training data
                for t in np.arange(1, max_timesteps+1):
                    # logging.critical("Timestep of episode: %0.2f" % self.sim.k)

                    # Perform one simulations step:
                    if not self.sim.stopped:
                        self.sim.step()  # Calls AcPolicy.customAction method.

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

                    if t % training_param["policy_rate"] == 0:
                        train_counter += 1
                        self.policy.agent.epsilon_decay_count = train_counter
                        self.policy.agent.timestep = np.int(t / training_param["policy_rate"])
                        if self.policy.agent.buffer.is_buffer_min_size():
                            model_update_counter += 1
                            if model_update_counter % training_param["model_update_rate"] == 0:
                                # TODO add data to episode buffer to get episode rewards while training.
                                mean_batch_reward, loss, td_error, grads, clipped_grads = self.policy.agent.train_step()
                                episode_mean_batch_rewards.append(mean_batch_reward)
                                episode_losses.append(loss)
                                episode_td_errors.append(td_error)
                                trained = True

                            if model_update_counter % training_param["target_update_rate"] == 0:
                                self.policy.agent.update_target_net()
                                # print("Updated target net.")

            reward = np.sum(episode_reward_list)/len(episode_reward_list)

            time_taken_episode = self.episodeTimer.endTime()

            if trained:
                if 0.05 * reward + (1 - 0.05) * running_reward > running_reward and episode_count % 50 == 0:
                    self.policy.agent.Q_actual_net.save_weights(model_param["weights_file_path"])
                    print("Saved network weights.")

                # Running reward smoothing effect
                running_reward = 0.05 * reward + (1 - 0.05) * running_reward


            if episode_count % 1 == 0 and trained:
                epsilon = self.policy.agent.calc_epsilon()
                if training_param["use_per"]:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Beta = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon,
                                                         self.policy.agent.buffer.beta, time_taken_episode)
                else:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon, time_taken_episode)

                print(print_output)
                logging.critical(print_output)
                # loss_list.append(loss)
                # episode_list.append(episode_count)
                # running_reward_list.append(running_reward)
                # training_var = (episode_list, running_reward_list, loss_list)

            # Save episode training variables to tensorboard
            tb_logger.save_histogram("Episode mean batch rewards", x=episode_count, y=episode_mean_batch_rewards)
            tb_logger.save_histogram("Episode losses", x=episode_count, y=episode_losses)
            tb_logger.save_histogram("Episode TD errors", x=episode_count, y=episode_td_errors)
            tb_logger.save_variable("Total episode reward (sum)", x=episode_count, y=np.sum(episode_reward_list))
            tb_logger.save_variable("Mean episode reward", x=episode_count, y=np.sum(episode_reward_list)/len(episode_reward_list))
            tb_logger.save_variable("Running reward", x=episode_count, y=running_reward)
            tb_logger.save_variable("Total time taken for episode", x=episode_count, y=time_taken_episode)
            tb_logger.save_variable("Mean vehicle speed for episode", x=episode_count, y=np.mean(vehicle_speeds))
            if training_param["use_per"]:
                tb_logger.save_variable("Beta increment", x=episode_count, y=self.policy.agent.buffer.beta)
            # TODO time taken for inferenece and time taken for training step

            # Save model weights and biases and gradients of backprop.
            # TODO fix deepset model so that we can save layer names
            tb_logger.save_weights_gradients(episode=episode_count,
                                             model=self.policy.agent.Q_actual_net,
                                             grads=grads,
                                             clipped_grads=clipped_grads)



                # self.data_logger.save_xls("./models/training_variables.xls")
            if running_reward >= training_param["final_return"] \
                    or episode_count == training_param["max_episodes"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                logging.critical(print_output)
                self.policy.agent.Q_actual_net.save_weights(model_param["weights_file_path"])
                # pic.dump(training_var, open("./models/train_output", "wb"))
                # self.data_logger.plot_training_data(plot_items)
                # self.data_logger.save_training_data("./models/training_variables.p")
                break

            episode_count += 1


    def simulate(self, simulation_timesteps):
        self.policy.agent.training = False
        self.policy.agent.evaluation = True
        steps = 0
        print_counter = 0
        with self.sim:
            self.create_plot()
            while not self.sim.stopped and not self.p.closed and steps<simulation_timesteps:
                self.sim.step()
                if print_counter % 10 == 0:
                    if self.policy.agent.latest_action == 0:
                        print("0: Slowing down.")
                    elif self.policy.agent.latest_action == 1:
                        print("1: Constant speed.")
                    elif self.policy.agent.latest_action == 2:
                        print("2: Speeding up.")
                    elif self.policy.agent.latest_action == 3:
                        print("3: Turning left.")
                    elif self.policy.agent.latest_action == 4:
                        print("4: Turning right.")
                steps += 1
                print_counter += 1
                self.p.plot()
            self.p.close()
            self.policy.agent.evaluation = False
            self.policy.agent.training = True


def sim_types(scenario_num, policy):
    # TODO make variation in sims
    # Randomised highway
    sim_config_0 = {
        "name": "AC_policy_dense_highway",
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
            {"amount": 16, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": 8, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": 4, "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    }

    # TODO Add additional randomness to fixedlane policy
    lane = random.randint(-1, 1)*3.6
    # Empty highway without cars
    sim_config_1 = {
        "name": "AC_policy_no_cars",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": policy, "R": 0, "l": lane, "s": 0, "v": np.random.random(1)*30.0, "D_MAX": 160},
            # {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": lane, "s": 30, "v": 28},
        ]
    }

    # Single car overtake
    sim_config_2 = {
        "name": "AC_policy_single_overtake",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"model": KBModel(), "policy": policy, "R": 0, "l": 0, "s": 0,
             "v": random.randint(25, 28), "D_MAX": 160},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 0, "s": 50, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 50, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 3.6, "s": 120, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 0, "s": 140, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 140, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 300, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 120, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 0, "s": 300, "v": 28}
        ]
    }

    # Double car overtake
    sim_config_3 = {
        "name": "AC_policy_double_overtake",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"model": KBModel(), "policy": policy, "R": 0, "l": 3.6, "s": 0,
             "v": random.randint(25, 28), "D_MAX": 160},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 40, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 3.6, "s": 50, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 150, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 125, "v": 24}
        ]
    }

    # Slow down car overtake right
    sim_config_4 = {
        "name": "AC_policy_slow_down_overtake_right",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"model": KBModel(), "policy": policy, "R": 0, "l": 3.6, "s": 0,
             "v": random.randint(25, 28), "D_MAX": 160},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 3.6, "s": 20, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 0, "v": 24}
        ]
    }

    # Slow down car overtake left
    sim_config_5 = {
        "name": "AC_policy_slow_down_overtake_left",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"model": KBModel(), "policy": policy, "R": 0, "l": -3.6, "s": 0,
             "v": random.randint(25, 28), "D_MAX": 160},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 20, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 10, "v": 24}
        ]
    }

    # Boxed in performance
    sim_config_6 = {
        "name": "AC_policy_boxed_in",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"model": KBModel(), "policy": policy, "R": 0, "l": 0, "s": 0,
             "v": random.randint(25, 28), "D_MAX": 160},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 20, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 20, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 3.6, "s": 20, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": -5, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 3.6, "s": 0, "v": 24},
        ]
    }

    sim_config = [
        sim_config_0,
        sim_config_1,
        sim_config_2,
        sim_config_3,
        sim_config_4,
        sim_config_5,
        sim_config_6
    ]

    return sim_config[scenario_num]

if __name__=="__main__":
    LOG_DIR = "logs"
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")
    config.scenarios_path = str(SC_PATH)

    # TODO check that hwsim config seed is set properly
    SEED = 10
    config.seed = SEED
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Logging
    logging.basicConfig(level=logging.INFO, filename="./logfiles/main.log")
    # logging.disable(logging.ERROR) # Temporarily disable error tf logs.
    with open('./logfiles/main.log', 'w'):
        pass  # Clear the log file of previous run

    # Model parameters:
    N_UNITS = (32, 16, 8)  # TODO Change model size variability between deepset and baseline
    N_INPUTS = 55
    N_ACTIONS = 5
    ACT_FUNC = tf.nn.selu
    N_STACKED_TIMESTEPS = 2
    MODEL_FILE_PATH = "./models/model_weights"
    model_param = {
        "n_units": N_UNITS,
        "n_inputs": N_INPUTS,
        "n_actions": N_ACTIONS,
        "activation_function": ACT_FUNC,
        "weights_file_path": MODEL_FILE_PATH,
        "seed": SEED,
        # TODO add parameters for the tuning of the deepset
        "cnn_param": {
            "config": 0,            # 0=1D conv. on vehicle dim.,
                                    # 1=1D conv. on measurements dim.,
                                    # 2=2D conv. on vehicle and measurements dimensions,
                                    # 3=3D conv. on vehicle and measurement dimensions through time
            # Config 0:
            "n_filters_0": 6,    # Dimensionality of output space
            "kernel_size_0": (2,),  # Convolution width
            # Config 1:
            "n_filters_1": 6,  # Dimensionality of output space
            "kernel_size_1": (2,),  # Convolution width
            # Config 2:
            "n_filters_2": 6,  # Dimensionality of output space
            "kernel_size_2": (4,2),  # Convolution width
            # Config 3:
            "n_filters_3": 6,  # Dimensionality of output space
            "n_timesteps": N_STACKED_TIMESTEPS,
            "kernel_size_3": (4,2,N_STACKED_TIMESTEPS)  # Convolution width
        }
        # TODO add initialiser
        # Add batch normalisation
    }
    logging.critical("Model Parameters:")
    logging.critical(model_param)
    pic.dump(model_param, open("./models/model_variables", "wb"))

    # Training parameters:
    POLICY_ACTION_RATE = 8     # Number of simulator steps before new control action is taken
    MAX_TIMESTEPS = 2.5e3         # range: 5e3 - 10e3
    MAX_EPISODES = 5000 #1.2e3
    FINAL_RETURN = 0.91
    SHOW_TRAIN_PLOTS = False
    SAVE_TRAINING = True
    LOG_FREQ = 0              # TODO implement log frequency
    PLOT_FREQ = 50
    SIM_TIMESTEPS = 200
    SCENARIO_NUM = 0        # 0-random_policies, 1-empty, 2-single_overtake, 3-double_overtake, etc.
    BUFFER_SIZE = 300000
    BATCH_SIZE = 32          # range: 32 - 150
    EPSILON_MIN = 1.0           # Exploration
    EPSILON_MAX = 0.1           # Exploitation
    DECAY_RATE = 0.9999 #0.999992
    MODEL_UPDATE_RATE = 1
    TARGET_UPDATE_RATE = 10e4
    LEARN_RATE = 0.0001         # range: 1e-3 - 1e-4
    OPTIMISER = tf.optimizers.Adam(learning_rate=LEARN_RATE)
    LOSS_FUNC = tf.losses.MeanSquaredError()  #tf.losses.Huber()  # PER loss function is MSE
    GAMMA = 0.99                # range: 0.95 - 0.99
    CLIP_GRADIENTS = False
    CLIP_NORM = 2
    # Reward weights = (rew_vel, rew_lat_lane_position, rew_fol_dist, staying_right, collision penalty)
    REWARD_WEIGHTS = np.array([1.0, 0.15, 0.8, 0.4, -5])
    STANDARDISE_RETURNS = True  # TODO additional variable for SPG
    USE_PER = False
    ALPHA = 0.75                # Priority scale: a=0:random, a=1:completely based on priority
    BETA = 0.2                  # Prioritisation factor
    BETA_INCREMENT = 0.00004 * MODEL_UPDATE_RATE    # Rate of Beta annealing to 1
    # Model types:
    USE_DUELLING = False
    USE_DEEPSET = False
    USE_CNN = True

    # TODO comparitive plotting of standard DQN, DDQN, PER, and Duelling
    # TODO plotting of average reward of vehicle that just speeds up
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
        "use_CNN": USE_CNN
    }
    logging.critical("Training param:")
    logging.critical(training_param)
    training_param_save = training_param.copy()
    training_param_save.pop("loss_func")
    training_param_save.pop("optimiser")
    pic.dump(training_param_save, open("./models/training_variables", "wb"))

    tb_logger = TbLogger(save_training=SAVE_TRAINING, seed=SEED, log_freq=LOG_FREQ)

    # Initialise models:
    # TODO Retrain on 3 network architectures
    # AC_net_single = AcNetworkSingleAction(model_param=model_param)
    # spg_agent_single = SpgAgentSingle(network=AC_net_single, training_param=training_param)
    # spg_policy_single = DiscreteSingleActionPolicy(agent=spg_agent_single)

    # TODO Fix Policies for double action as well as single action
    # AC_net_double = AcNetworkDoubleAction(model_param=model_param)
    # spg_agent_double = SpgAgentDouble(network=AC_net_double, training_param=training_param)
    # spg_policy_double = DiscreteDoubleActionPolicy(agent=spg_agent_double)

    # TODO Compare DQN DDQN PER and DUELLING ON SAME RANDOM SEED!
    # if USE_DUELLING:
    #     DQ_net = DuellingDqnNetwork(model_param=model_param)
    # else:
    #     DQ_net = DeepQNetwork(model_param=model_param)

    if USE_DEEPSET and not USE_CNN:
        DQ_net = DeepSetQNetwork(model_param=model_param)
    elif not USE_DEEPSET and USE_CNN:
        DQ_net = CNN(model_param=model_param)
    elif not USE_DEEPSET and not USE_CNN:
        DQ_net = DeepQNetwork(model_param=model_param)
    else:
        print("Error: Cannot use Deepset and CNN methods together!")
        exit()

    dqn_agent = DqnAgent(network=DQ_net, training_param=training_param, tb_logger=tb_logger)
    dqn_policy = DiscreteSingleActionPolicy(agent=dqn_agent)

    RewardFunction().plot_reward_functions()

    # Set up main class for running simulations:
    main = Main(scenario_num=SCENARIO_NUM, policy=dqn_policy)
    time.sleep(0.01)
    # main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
    main.policy.agent.evaluation = False
    # Train model:

    # TODO Ensure Dmax is consistently 150 when merging the branch with master!!!!!
    main.train_policy()  # TODO !!!

    # TODO Tidy up simulation part:
    # Simulate model:
    main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
    main.policy.agent.evaluation = True
    main.simulate(1000)
    # for i in range(0, 5):
    #     for _ in range(2):
    #         main = Main(sim_types(i))
    #         main.simulate(1000)
    #         # print("Simulation number %d complete" % i)
    #         main.p.close()

    print("EOF")


