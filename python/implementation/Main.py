import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
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
        self.trainerTimer = Timer("the Trainer")
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
        episode_list = []
        running_reward_list = []
        loss_list = []
        # plot_items = self.data_logger.init_training_plot()
        train_counter = 1
        model_update_counter = 1
        # Run until all episodes are completed (reward reached).
        while episode_count <= max_episodes:
            self.policy.agent.episode = episode_count

            if episode_count % plot_freq == 0 and show_plots:
                self.policy.agent.prev_epsilon = self.policy.agent.epsilon
                self.simulate(training_param["sim_timesteps"])
                self.policy.agent.epsilon = self.policy.agent.prev_epsilon

            # print("Episode number: %0.2f" % episode_count)
            logging.critical("Episode number: %0.2f" % episode_count)

            # Set simulation environment
            with self.sim:
                # Loop through each timestep in episode.
                with self.episodeTimer:
                    # Run the model for one episode to collect training data
                    # Saves actions values, critic values, and rewards in policy class variables
                    for t in np.arange(1, max_timesteps+1):
                        # logging.critical("Timestep of episode: %0.2f" % self.sim.k)

                        if t%training_param["policy_rate"] == 0:
                            train_counter += 1
                            self.policy.agent.epsilon_decay_count = train_counter
                            self.policy.agent.timestep = np.int(t/training_param["policy_rate"])
                            if self.policy.agent.buffer.is_buffer_min_size():
                                model_update_counter += 1
                                if model_update_counter % training_param["model_update_rate"] == 0:
                                    reward, loss = self.policy.agent.train_step()
                                    trained = True
                                if model_update_counter % training_param["target_update_rate"] == 0:
                                    self.policy.agent.update_target_net()
                                    print("Updated target net.")

                        self.policy.agent.stop_flags = self.sim.stopped or self.sim._collision
                        if self.policy.agent.stop_flags == True:
                            self.policy.agent.buffer.alter_buffer_stop_flag(flag=self.policy.agent.stop_flags)

                        # Perform one simulations step:
                        if not self.sim.stopped:
                            self.sim.step()  # Calls AcPolicy.customAction method.

                            # if self.sim._collision:
                            #     logging.critical("Collision. At episode %f" % episode_count)
                            #     policy.trainer.set_neg_collision_reward(np.int(t/training_param["STEP_TIME"]),
                            #                                             training_param["reward_weights"][4])

                # with trainerTimer:
                #     policy.trainer.buffer.set_tf_experience_for_episode_training()
                #     episode_reward, loss = policy.trainer.train_step()

                # self.data_logger.set_complete_episode(policy.trainer.buffer.get_experience())

                # Clear loss values and reward history
                # policy.trainer.buffer.clear_experience()

            if trained:
                if 0.05 * reward + (1 - 0.05) * running_reward > running_reward and episode_count%50==0:
                    self.policy.agent.Q_actual_net.save_weights(model_param["weights_file_path"])
                    print("Saved network weights.")

                # Running reward smoothing effect
                running_reward = 0.05 * reward + (1 - 0.05) * running_reward


            if episode_count % 5 == 0 and trained:
                epsilon = self.policy.agent.calc_epsilon()
                print_template = "Running reward = {:.2f} at episode {}. Loss = {:.2f}. Epsilon = {:.2f}."
                print_output = print_template.format(running_reward, episode_count, loss, epsilon)
                loss_list.append(loss)
                episode_list.append(episode_count)
                running_reward_list.append(running_reward)
                training_var = (episode_list, running_reward_list, loss_list)
                print(print_output)
                logging.critical(print_output)

                # self.data_logger.save_xls("./models/training_variables.xls")
            if running_reward >= training_param["final_return"] \
                    or episode_count == training_param["max_episodes"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                logging.critical(print_output)
                self.policy.agent.Q_actual_net.save_weights(model_param["weights_file_path"])
                pic.dump(training_var, open("./models/train_output", "wb"))
                # self.data_logger.plot_training_data(plot_items)
                # self.data_logger.save_training_data("./models/training_variables.p")
                break

            episode_count += 1

            # if episode_count % plot_freq == 0 and show_plots:
            #     self.data_logger.plot_training_data(plot_items)



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
            {"amount": 1, "model": KBModel(), "policy": policy},
            # {"amount": 2, "model": KBModel(), "policy": StepPolicy(10, [0.1, 0.5])},
            # {"amount": 1, "model": KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            # {"amount": 8, "model": KBModel(), "policy": IMPolicy()},
            {"amount": 30, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": 20, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": 10, "model": KBModel(), "policy": BasicPolicy("fast")}
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
            {"amount": 1, "model": KBModel(), "policy": policy, "R": 0, "l": lane, "s": 0, "v": np.random.random(1)*30.0},
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
             "v": random.randint(25, 28)},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 0, "s": 50, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 50, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 3.6, "s": 120, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": 0, "s": 140, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 140, "v": 28},
            {"model": KBModel(), "policy": FixedLanePolicy(28), "R": 0, "l": -3.6, "s": 300, "v": 28},            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 120, "v": 24},
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
             "v": random.randint(25, 28)},
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
             "v": random.randint(25, 28)},
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
             "v": random.randint(25, 28)},
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
             "v": random.randint(25, 28)},
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

    # Logging
    logging.basicConfig(level=logging.INFO, filename="./logfiles/main.log")
    # logging.disable(logging.ERROR) # Temporarily disable error tf logs.
    with open('./logfiles/main.log', 'w'):
        pass  # Clear the log file of previous run

    SEED = config.seed
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Model parameters:
    N_UNITS = (64, 32, 16)
    N_INPUTS = 54
    N_ACTIONS = 5
    ACT_FUNC = tf.nn.relu
    MODEL_FILE_PATH = "./models/model_weights"
    model_param = {
        "n_units": N_UNITS,
        "n_inputs": N_INPUTS,
        "n_actions": N_ACTIONS,
        "activation_function": ACT_FUNC,
        "weights_file_path": MODEL_FILE_PATH,
        "seed": SEED
    }
    logging.critical("Model Parameters:")
    logging.critical(model_param)
    pic.dump(model_param, open("./models/model_variables", "wb"))

    # Training parameters:
    POLICY_ACTION_RATE = 10     # Number of simulator steps before new control action is taken
    MAX_TIMESTEPS = 3e3         # range: 5e3 - 10e3
    MAX_EPISODES = 3e3
    FINAL_RETURN = 1e10
    SHOW_TRAIN_PLOTS = False
    PLOT_FREQ = 50
    SIM_TIMESTEPS = 100
    SCENARIO_NUM = 0            # 0-random_policies, 1-empty, 2-single_overtake, 3-double_overtake, etc.
    BUFFER_SIZE = 300000
    BATCH_SIZE = 250       # range: 32 - 150
    EPSILON_MIN = 1.0           # Exploration
    EPSILON_MAX = 0.1           # Exploitation
    DECAY_RATE = 0.999995
    MODEL_UPDATE_RATE = 400
    TARGET_UPDATE_RATE = 50*MODEL_UPDATE_RATE
    LEARN_RATE = 0.0005         # range: 1e-3 - 1e-4
    OPTIMISER = tf.optimizers.Adam(learning_rate=LEARN_RATE)
    LOSS_FUNC = tf.losses.Huber()
    GAMMA = 0.99                # range: 0.95 - 0.99
    CLIP_GRADIENTS = True
    CLIP_NORM = 2
    # Reward weights = (rew_vel, rew_lat_lane_position, rew_fol_dist, staying_right, collision penalty)
    REWARD_WEIGHTS = np.array([1.0, 0.15, 1.0, 0.3, -5])
    STANDARDISE_RETURNS = True
    training_param = {
        "max_timesteps": MAX_TIMESTEPS,
        "max_episodes": MAX_EPISODES,
        "final_return": FINAL_RETURN,
        "show_train_plots": SHOW_TRAIN_PLOTS,
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
        "reward_weights": REWARD_WEIGHTS
    }
    logging.critical("Training param:")
    logging.critical(training_param)
    training_param_save = training_param.copy()
    training_param_save.pop("loss_func")
    training_param_save.pop("optimiser")
    pic.dump(training_param_save, open("./models/training_variables", "wb"))

    # Initialise models:
    # TODO Retrain on 3 network architectures
    AC_net_single = AcNetworkSingleAction(model_param=model_param)
    spg_agent_single = SpgAgentSingle(network=AC_net_single, training_param=training_param)
    spg_policy_single = DiscreteSingleActionPolicy(agent=spg_agent_single)

    # TODO Fix Policies for double action as well as single action
    # AC_net_double = AcNetworkDoubleAction(model_param=model_param)
    # spg_agent_double = SpgAgentDouble(network=AC_net_double, training_param=training_param)
    # spg_policy_double = DiscreteDoubleActionPolicy(agent=spg_agent_double)

    DQ_net = DeepQNetwork(model_param=model_param)
    dqn_agent = DqnAgent(network=DQ_net, training_param=training_param)
    dqn_policy = DiscreteSingleActionPolicy(agent=dqn_agent)

    RewardFunction().plot_reward_functions()

    # Set up main class for running simulations:
    main = Main(scenario_num=SCENARIO_NUM, policy=dqn_policy)
    time.sleep(0.01)
    # main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
    main.policy.agent.evaluation = False
    # Train model:
    # main.train_policy()

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


