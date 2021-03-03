import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

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

# tf.config.experimental.set_visible_devices([0], "GPU")

class Main(object):

    def __init__(self, sim_config):
        # ...
        self.sim = Simulation(sim_config)
        self.pol = [{  # List of autonomous policies
            "ids": [0],  # Vehicle IDs of vehicles equipped with this autonomous policy
            "policy": sim_config["vehicles"][0]["policy"],  # Reference to the policy
            "metrics": {}  # Extra metrics we calculate for each autonomous policy
        }]
        # Create object for data logging and visualisation
        self.data_logger = DataLogger(model_param, training_param)

    # ...
    def create_plot(self):
        shape = (4, 2)
        groups = [([0, 2], 0)]
        vehicle_type = "car" if FANCY_CARS else "cuboid3D"
        # self.p = Plotter(self.sim, "Multi car simulation", mode=PLOT_MODE, shape=shape, groups=groups, off_screen=OFF_SCREEN)
        self.p = Plotter(self.sim, "Multi car simulation", mode=Plotter.Mode.MP4, shape=shape, groups=groups,
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
        # self.p.subplot(3, 0)
        # self.p.add_text("Rewards")
        # self.p.TimeChartPlot(self.p, lines=self.pol[0]["policy"].buffer.rewards)
        self.p.plot()  # Initial plot

    def train_policy(self):
        running_reward = 0
        episode_count = 1
        plot_freq = training_param["plot_freq"]
        show_plots = training_param["show_plots_when_training"]
        max_timesteps_episode = training_param["max_steps_per_episode"]
        policy = self.pol[0]["policy"]
        trained = False
        # plot_items = self.data_logger.init_training_plot()
        train_counter = 1
        model_update_counter = 1
        # Run until all episodes are completed (reward reached).
        while episode_count <= training_param["max_episodes"]:
            policy.trainer.episode = episode_count

            if episode_count % plot_freq == 0 and show_plots:
                self.simulate(training_param["simulation_timesteps"])

            # print("Episode number: %0.2f" % episode_count)
            logging.critical("Episode number: %0.2f" % episode_count)

            # Set simulation environment
            with self.sim:
                # Loop through each timestep in episode.
                with episodeTimer:
                    # Run the model for one episode to collect training data
                    # Saves actions values, critic values, and rewards in policy class variables
                    for t in np.arange(1, max_timesteps_episode+1):
                        # logging.critical("Timestep of episode: %0.2f" % self.sim.k)

                        trainer.stop_flags = self.sim.stopped
                        if t%training_param["STEP_TIME"] == 0:
                            train_counter += 1
                            trainer.epsilon_decay_count = train_counter
                            trainer.timestep = np.int(t/training_param["STEP_TIME"])
                            if trainer.buffer.is_buffer_min_size():
                                train_counter += 1
                                model_update_counter += 1
                                if model_update_counter % training_param["model_update_rate"] == 0:
                                    reward, loss = trainer.train_step()
                                    trained = True
                                if model_update_counter % training_param["target_update_rate"] == 0:
                                    trainer.update_target_net()
                                    print("Updated target net.")



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
                # Running reward smoothing effect
                running_reward = 0.05 * reward + (1 - 0.05) * running_reward

                if 0.05 * reward + (1 - 0.05) * running_reward > running_reward:
                    policy.trainer.Q_actual_net.save_weights(model_param["weights_file_path"])

            if episode_count % 5 == 0 and trained:
                epsilon = trainer.calc_epsilon()
                print_template = "Running reward = {:.2f} at episode {}. Loss = {:.2f}. Epsilon = {:.2f}."
                print_output = print_template.format(running_reward, episode_count, loss, epsilon)
                print(print_output)
                logging.critical(print_output)

                # self.data_logger.save_xls("./trained_models/training_variables.xls")
            if running_reward >= training_param["final_return"] \
                    or episode_count == training_param["max_episodes"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                logging.critical(print_output)
                policy.trainer.Q_actual_net.save_weights(model_param["weights_file_path"])
                # self.data_logger.plot_training_data(plot_items)
                # self.data_logger.save_training_data("./trained_models/training_variables.p")


                break

            episode_count += 1

            # if episode_count % plot_freq == 0 and show_plots:
            #     self.data_logger.plot_training_data(plot_items)



    def simulate(self, simulation_timesteps):
        self.pol[0]["policy"].trainer.training = False
        with self.sim:
            self.create_plot()
            t = 1
            while not self.sim.stopped:
                self.sim.step()
                self.p.plot()
                t += 1
                if t == simulation_timesteps:
                    self.p.close()
                    break

        self.pol[0]["policy"].trainer.training = True

def sim_types(sim_num):
    # Randomised highway
    sim_config_0 = {
        "name": "AC_policy_dense_highway",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": DiscreteActionPolicy(trainer)},
            # {"amount": 2, "model": KBModel(), "policy": StepPolicy(10, [0.1, 0.5])},
            # {"amount": 1, "model": KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            # {"amount": 8, "model": KBModel(), "policy": IMPolicy()},
            {"amount": 22, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": 14, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": 7, "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    }

    # TODO Add additional randomness to fixedlane policy

    # Empty highway without cars
    sim_config_1 = {
        "name": "AC_policy_no_cars",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": DiscreteActionPolicy(trainer)}
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
            {"model": KBModel(), "policy": DiscreteActionPolicy(trainer), "R": 0, "l": 0, "s": 0,
             "v": random.randint(25, 28)},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 50, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 3.6, "s": 150, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": 0, "s": 300, "v": 24},
            {"model": KBModel(), "policy": FixedLanePolicy(24), "R": 0, "l": -3.6, "s": 450, "v": 24}
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
            {"model": KBModel(), "policy": DiscreteActionPolicy(trainer), "R": 0, "l": 3.6, "s": 0,
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
            {"model": KBModel(), "policy": DiscreteActionPolicy(trainer), "R": 0, "l": 3.6, "s": 0,
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
            {"model": KBModel(), "policy": DiscreteActionPolicy(trainer), "R": 0, "l": -3.6, "s": 0,
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
            {"model": KBModel(), "policy": DiscreteActionPolicy(trainer), "R": 0, "l": 0, "s": 0,
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

    return sim_config[sim_num]

if __name__=="__main__":
    # Initial configuration
    ID = -1  # ID of simulation to replay or -1 to create a new one
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    FANCY_CARS = False
    LOG_DIR = "logs"
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")

    # Logging
    logging.basicConfig(level=logging.INFO, filename="./logfiles/main.log")
    # TODO Enable TF warnings and query with Bram

    logging.disable(logging.ERROR) # Temporarily disable error tf logs.

    with open('./logfiles/main.log', 'w'):
        pass  # Clear the log file of previous run

    config.scenarios_path = str(SC_PATH)
    print_output = "Using seed %f" % config.seed
    print(print_output)
    logging.critical(print_output)

    seed = config.seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Model configuration and settings
    model_param = {
        "n_units": (100, 50),
        "n_inputs": 54,  # Standard size of S
        "activation_function": tf.nn.relu,  # activation function of hidden nodes
        "n_actions": 2,
        "weights_file_path": "./trained_models/model_weights",
        "trained_model_file_path": "./trained_models/trained_model",
        "seed": seed
    }
    logging.critical("Model Parameters:")
    logging.critical(model_param)

    STEP_TIME = 10
    optimiser = "ADAM"
    learning_rate = 0.0001
    if optimiser == "ADAM":
        optimiser_name = optimiser
        optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimiser == "RMS":
        optimiser_name = optimiser
        optimiser = tf.optimizers.RMSprop(learning_rate=learning_rate)

    training_param = {
        "max_steps_per_episode": 3e3,
        "max_episodes": 2200,
        "final_return": 1e10,
        "show_plots_when_training": True,
        "plot_freq": 50,
        "simulation_timesteps": 200,
        "max_buffer_size": 100000,
        "batch_size": 400,
        "evaluation": False,
        "epsilon_max": 1.0,  # Initial epsilon - Exploration
        "epsilon_min": 0.1,  # Final epsilon - Exploitation
        "decay_rate": 0.99999,
        "model_update_rate": 400,
        "target_update_rate": 1e4,
        "STEP_TIME": STEP_TIME,  # Currently not implemented
        "gamma": 0.95,
        "clip_gradients": True,
        "clip_norm": 2,
        "standardise_returns": True,
        "learning_rate": learning_rate,
        "optimiser_name": optimiser_name,
        "optimiser": optimiser,
        "loss_func": tf.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
        "seed": seed,
        "reward_weights": np.array([1.1, 0., 0., 0.6, -5])  # (rew_vel, rew_lat_position, rew_fol_dist, collision penalty)
    }
    logging.critical("Training param:")
    logging.critical(training_param)

    # Initialise network/model architecture:
    # actor_critic_net = ActorCriticNetDiscrete(model_param)
    # trainer = GradAscentTrainerDiscrete(actor_critic_net, training_param)
    DQ_net = DeepQNetwork(model_param)
    trainer = DqnTrainer(network=DQ_net, training_param=training_param)


    # Simulation configuration and settings
    # TODO Move to randomly initialising vehicles infront of ego vehicle to learn more complex actions than staying in lane
    safetyCfg = {
        "Mvel": 1.0,
        "Gth": 2.0
    }

    plotTimer = Timer("Plotting")
    trainerTimer = Timer("the Trainer")
    episodeTimer = Timer("Episode")

    sim_number = 0
    sim = Simulation(sim_types(sim_number))

    # Set up main class for running simulations:
    main = Main(sim_types(sim_number))

    # Train model:
    main.train_policy()

    # Simulate model:
    main.pol[0]["policy"].trainer.Q_actual_net.load_weights(model_param["weights_file_path"])
    for i in range(0, 5):
        main = Main(sim_types(2))
        main.simulate(1000)
        # print("Simulation number %d complete" % i)
        main.p.close()

    print("EOF")


