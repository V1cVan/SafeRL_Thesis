from hwsim import Simulation, BasicPolicy, StepPolicy, SwayPolicy, IMPolicy, KBModel, TrackPolicy, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot

import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from NeuralModels import *
from RL_Policies import *
from HelperClasses import *
import logging

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

class Main(object):

    def __init__(self, sim_config):
        # ...
        self.sim = Simulation(sim_config)
        self.pol = [{  # List of autonomous policies
            "ids": [0],  # Vehicle IDs of vehicles equipped with this autonomous policy
            "policy": sim_config["vehicles"][0]["policy"],  # Reference to the policy
            "metrics": {}  # Extra metrics we calculate for each autonomous policy
        }]

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
        self.p.subplot(3, 0)
        self.p.add_text("Rewards")
        # self.p.TimeChartPlot(self.p, lines=self.pol[0]["policy"].buffer.rewards)
        self.p.plot()  # Initial plot


    def train_policy(self):
        running_reward = 0
        episode_count = 1
        plot_freq = training_param["plot_freq"]
        show_plots = training_param["show_plots_when_training"]
        max_timesteps_episode = training_param["max_steps_per_episode"]
        policy = self.pol[0]["policy"]

        # Run until all episodes are completed (reward reached).
        while True:
            if episode_count % plot_freq == 0 and show_plots:
                self.simulate(training_param["simulation_timesteps"])

            logging.critical("Episode number: %0.2f" % episode_count)

            # Set simulation environment
            with self.sim:
                # Loop through each timestep in episode.
                with episodeTimer:
                    # Run the model for one episode to collect training data
                    # Saves actions values, critic values, and rewards in policy class variables
                    for t in np.arange(1, max_timesteps_episode+1):
                        # logging.critical("Timestep of episode: %0.2f" % self.sim.k)
                        policy.trainer.timestep = t
                        # Perform one simulations step:
                        if not self.sim.stopped:
                            self.sim.step()  # Calls AcPolicy.customAction method.
                            if self.sim._collision:
                                logging.critical("Collision. At episode %f" % episode_count)
                                #policy.trainer.set_neg_collision_reward(t, -1)
                                break

                # Batch policy update
                with trainerTimer:
                    policy.trainer.buffer.set_tf_experience_for_episode_training()
                    episode_reward = policy.trainer.train_step()

                    # Clear loss values and reward history
                    policy.trainer.buffer.clear_experience()

            # Running reward smoothing effect
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            episode_count += 1
            if episode_count % 5 == 0:
                print_template = "Running reward = {:.2f} at episode {}"
                print_output = print_template.format(running_reward, episode_count)
                print(print_output)
                logging.critical(print_output)
            if running_reward >= training_param["final_return"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                logging.critical(print_output)
                policy.trainer.actor_critic_net.save_weights(model_param["weights_file_path"])
                break

            # Save intermediate policies
            if episode_count % 100 == 0:
                policy.trainer.actor_critic_net.save_weights(model_param["weights_file_path"])

    def simulate(self, simulation_timesteps):
        self.pol[0]["policy"].trainer.training = False
        with self.sim:
            self.create_plot()
            t = 1
            while not self.sim.stopped and not self.p.closed:
                self.sim.step()
                self.p.plot()
                t += 1
                if t == simulation_timesteps:
                    self.p.close()

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
            {"amount": 1, "model": KBModel(), "policy": AcPolicyDiscrete(trainer)},
            {"amount": 2, "model": KBModel(), "policy": StepPolicy(10, [0.1, 0.5])},
            {"amount": 1, "model": KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            {"amount": 8, "model": KBModel(), "policy": IMPolicy()},
            {"amount": 10, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": 18, "model": KBModel(), "policy": BasicPolicy("normal")},
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
            {"amount": 1, "model": KBModel(), "policy": AcPolicyDiscrete(trainer)}
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
            {"model": KBModel(), "policy": AcPolicyDiscrete(trainer), "R": 0, "l": 0, "s": 0,
             "v": random.randint(25,28)},
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
            {"model": KBModel(), "policy": AcPolicyDiscrete(trainer), "R": 0, "l": 3.6, "s": 0,
             "v": random.randint(25,28)},
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
            {"model": KBModel(), "policy": AcPolicyDiscrete(trainer), "R": 0, "l": 3.6, "s": 0,
             "v": random.randint(25,28)},
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
            {"model": KBModel(), "policy": AcPolicyDiscrete(trainer), "R": 0, "l": -3.6, "s": 0,
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
            {"model": KBModel(), "policy": AcPolicyDiscrete(trainer), "R": 0, "l": 0, "s": 0,
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
    PLOT_MODE = Plotter.Mode.MP4
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
        "n_nodes": [160, 80],  # Number of hidden nodes in each layer
        "n_layers": 2,  # Number of layers
        "n_inputs": 54,  # Standard size of S
        "activation_function": tf.nn.relu,  # activation function of hidden nodes
        "n_actions": 2,
        "weights_file_path": "./python/implementation/trained_models/model_weights_overnight",
        "trained_model_file_path": "./python/implementation/trained_models/trained_model",
        "seed": seed
    }
    logging.critical("Model Parameters:")
    logging.critical(model_param)
    training_param = {
        "max_steps_per_episode":  00,
        "final_return": 1000,
        "show_plots_when_training": False,
        "plot_freq": 10,
        "simulation_timesteps": 1000,
        "STEP_TIME": 10,  # Currently not implemented
        "gamma": 0.99,  # Discount factor
        "adam_optimiser": keras.optimizers.Adam(learning_rate=0.0005),
        "huber_loss": keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
        "seed": seed,
        "reward_weights": np.array([0.5, 0, 0.5])  # (rew_vel, rew_lat_position, rew_fol_dist)
    }
    logging.critical("Training param:")
    logging.critical(training_param)

    # Initialise network/model architecture:
    actor_critic_net = ActorCriticNetDiscrete(model_param)
    actor_critic_net.display_overview()
    trainer = GradAscentTrainerDiscrete(actor_critic_net, training_param)  # training method used

    # Simulation configuration and settings
    # TODO Move to randomly initialising vehicles infront of ego vehicle to learn more complex actions than staying in lane
    safetyCfg = {
        "Mvel": 1.0,
        "Gth": 2.0
    }

    plotTimer = Timer("Plotting")
    trainerTimer = Timer("the Trainer")
    episodeTimer = Timer("Episode")

    # Create object for data logging and visualisation
    data_logger = DataLogger(seed, model_param, training_param)

    sim = Simulation(sim_types(0))

    # Set up main class for running simulations:
    main = Main(sim_types(0))

    # Train model:
    # main.train_policy()

    # Simulate model:
    main.pol[0]["policy"].trainer.actor_critic_net.load_weights(model_param["weights_file_path"])
    for i in range(1,5):
        main = Main(sim_types(6))
        main.simulate(6000)
        # print("Simulation number %d complete" % i)
        # main.p.close()

    print("EOF")


