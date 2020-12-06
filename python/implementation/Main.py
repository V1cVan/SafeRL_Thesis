from hwsim import Simulation, BasicPolicy, KBModel, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot

import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from NeuralModels import *
from RL_Policies import *
from RL_Policies import *
from HelperClasses import *
import logging

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

class Main(object):

    def __init__(self, sim_config):
        # ...
        self.sim = Simulation(sim_config)
        self.pol = [{  # List of autonomous policies
            "ids": [0],  # Vehicle IDs of vehicles equiped with this autonomous policy
            "policy": sim_config["vehicles"][0]["policy"],  # Reference to the policy
            "metrics": {}  # Extra metrics we calculate for each autonomous policy
        }]

    # ...
    def create_plot(self):
        shape = (4, 2)
        groups = [([0, 2], 0)]
        vehicle_type = "car" if FANCY_CARS else "cuboid3D"
        self.p = Plotter(self.sim, "Multi car simulation", mode=PLOT_MODE, shape=shape, groups=groups, off_screen=OFF_SCREEN)
        self.p.V = 0
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
        self.p.subplot(2,1)
        self.p.add_text("Actions")
        ActionsPlot(self.p,actions="long")
        self.p.subplot(3,1)
        ActionsPlot(self.p,actions="lat")
        self.p.plot()  # Initial plot

    # TODO query tf.function problems with Bram!
    # TODO query tf.numpy_function of of sim.step?
    #tf.function
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
                    for t in tf.range(1, max_timesteps_episode+1):
                        # logging.critical("Timestep of episode: %0.2f" % self.sim.k)
                        policy.trainer.timestep = t
                        # Perform one simulations step:
                        if not self.sim.stopped:
                            self.sim.step()  # Calls AcPolicy.customAction method.
                            if self.sim._collision:
                                logging.critical("Collision. At episode %f" % episode_count)
                                policy.trainer.set_neg_collision_reward(t, -5)
                                break

                # Batch policy update
                with trainerTimer:
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
            if episode_count % 50 == 0:
                policy.trainer.actor_critic_net.save_weights(model_param["weights_file_path"])

    def simulate(self, simulation_timesteps):
        self.pol[0]["policy"].trainer.training = False
        with self.sim:
            self.create_plot()
            for t in np.arange(simulation_timesteps):
                if not self.sim.stopped and not self.p.closed:
                    self.sim.step()
                    self.p.plot()
            self.p.close()
        self.pol[0]["policy"].trainer.training = True

if __name__=="__main__":
    # Initial configuration
    ID = -1  # ID of simulation to replay or -1 to create a new one
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    FANCY_CARS = True
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
        "n_nodes": [400, 400],  # Number of hidden nodes in each layer
        "n_layers": 2,  # Number of layers
        "n_inputs": 30,  # Standard size of S
        "n_actions": 2,
        "weights_file_path": "./python/implementation/trained_models/model_weights",
        "seed": seed
    }
    logging.critical("Model Parameters:")
    logging.critical(model_param)
    training_param = {
        "max_steps_per_episode": 300,  # TODO kM - max value of k
        "final_return": 1000,
        "show_plots_when_training": True,
        "plot_freq": 5,  # TODO Reimplement plot freq (debug why crash)
        "simulation_timesteps": 100,
        "gamma": 0.99,  # Discount factor
        # TODO Check results of different learning rates
        "adam_optimiser": keras.optimizers.Adam(learning_rate=0.01),
        # TODO Check results of different loss functions sum/mse
        "huber_loss": keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
        "seed": seed,
        "reward_weights": np.array([0.3, 0.3, 0.3])  # (rew_vel, rew_lat_position, rew_fol_dist)
    }
    logging.critical("Training param:")
    logging.critical(training_param)

    # Initialise network/model architecture:
    actor_critic_net = ActorCriticNetDiscrete(model_param)
    actor_critic_net.display_overview()
    trainer = GradAscentTrainerDiscrete(actor_critic_net, training_param)  # training method used

    # Simulation configuration and settings
    # TODO Move to training on more complex scenario without other vehicles.
    veh_types = [
        {"amount": 1, "model": KBModel(), "policy": AcPolicyDiscrete(trainer)},
        {"amount": 80, "model": KBModel(), "policy": BasicPolicy("slow")},
        {"amount": 15, "model": KBModel(), "policy": BasicPolicy("normal")},
        {"amount": 10, "model": KBModel(), "policy": BasicPolicy("fast")}
    ]
    # veh_types = [
    #     {"amount": 1, "model": KBModel(), "policy": AcPolicyDiscrete(trainer)}
    # ]
    sim_config = {
        "name": "AC_policy",
        "scenario": "CIRCUIT",
        #"kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": veh_types
    }

    plotTimer = Timer("Plotting")
    trainerTimer = Timer("the Trainer")
    episodeTimer = Timer("Episode")

    # Create object for data logging and visualisation
    data_logger = DataLogger(seed, model_param, training_param)

    sim = Simulation(sim_config)

    # Set up main class for running simulations:
    main = Main(sim_config)

    # Train model:
    main.train_policy()

    # Simulate model:
    main.pol[0]["policy"].trainer.actor_critic_net.load_weights(model_param["weights_file_path"])
    main.simulate(training_param["simulation_timesteps"])

    print("EOF")


