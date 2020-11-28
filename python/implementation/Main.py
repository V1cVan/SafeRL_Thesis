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
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
        ActionsPlot(self.p,actions="vel")
        self.p.subplot(3,1)
        ActionsPlot(self.p,actions="off")
        self.p.plot()  # Initial plot

    # TODO query tf.function problems with Bram!
    # TODO query tf.numpy_function of of sim.step?
    #tf.function
    def train_policy(self):
        running_reward = 0
        episode_count = 0
        policy = self.pol[0]["policy"]

        # Run until all episodes are completed (reward reached).
        while True:
            logging.info(" - - Episode number: %0.2f" % episode_count)
            # Set simulation environment
            with self.sim:
                self.create_plot()
                episode_reward = 0
                timestep = 0
                # Loop through each timestep in episode.
                # TODO Explain use of gradienttape not working when used here on sim.step
                with episodeTimer:
                    # Run the model for one episode to collect training data
                    # Saves actions values, critic values, and rewards in policy class variables
                    for t in tf.range(training_param["max_steps_per_episode"]):
                        logging.info(" - Timestep of episode: %0.2f" % self.sim.k)
                        policy.trainer.set_timestep(t)
                        # Perform one simulations step:
                        self.sim.step()  # Calls AcPolicy.customAction method.
                        with plotTimer:
                            self.p.plot()
                        if sim.stopped:
                            break

                    # Batch policy update
                    with trainerTimer:
                        states, actions_vel, actions_off, action_vel_choice, action_off_choice, rewards = policy.trainer.get_experience()
                        policy.trainer.set_tf_action_choices(states, actions_vel, actions_off, action_vel_choice, action_off_choice, rewards)
                        episode_reward = policy.trainer.train_step()

                # Running reward smoothing effect
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            episode_count += 1
            if episode_count % 10 == 0:
                print_template = "Running reward = {:.2f} at episode {}"
                print(print_template.format(running_reward, episode_count))
            if running_reward >= final_return:
                print("Solved at episode {}!".format(episode_count))
                break

    def simulate(self):
        with self.sim:
            while not self.sim.stopped:
                # Perform one simulation step:
                self.sim.step()
                # ... (visualization/extra callbacks)

                # Keep track of average metrics in this episode
                # ...

                # Perform one train step, using the collected new experience:
                for auto in self.auto:
                    if auto["policy"].trainer.training:
                        auto["policy"].trainer.train_step()


if __name__=="__main__":
    # Initial configuration
    ID = -1 # ID of simulation to replay or -1 to create a new one
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    FANCY_CARS = True
    LOG_DIR = "logs"
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    # config.seed = 1249517370
    print(f"Using seed {config.seed}")

    # Logging
    logging.basicConfig(level=logging.INFO, filename="./python/implementation/logfiles/main.log")
    with open('./python/implementation/logfiles/main.log', 'w'):
        pass  # Clear the log file of previous run

    # Model configuration and settings
    model_param = {
        "n_nodes": [400, 200],  # Number of hidden nodes in each layer
        "n_layers": 2,  # Number of layers
        "n_inputs": 47,  # Standard size of S
        "n_actions": 2
    }
    training_param = {
        "max_steps_per_episode": 10,  # TODO kM - max value of k
        "final_return": 150,
        "gamma": 0.99,  # Discount factor
        "adam_optimiser": keras.optimizers.Adam(learning_rate=0.01),
        "huber_loss": keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    }

    # Initialise network/model architecture:
    actor_critic_net = ActorCriticNetDiscrete(model_param)
    actor_critic_net.display_overview()
    trainer = GradAscentTrainerDiscrete(actor_critic_net, training_param)  # training method used

    # Simulation configuration and settings
    veh_types = [
        {"amount": 1, "model": KBModel(), "policy": AcPolicyDiscrete(trainer)},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("slow")},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("normal")},
        {"amount": 20, "model": KBModel(), "policy": BasicPolicy("fast")}
    ]
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

    sim = Simulation(sim_config)

    # Set up main class for running simulations:
    main = Main(sim_config)

    # Train model:
    main.train_policy()

    # Simulate model:


    print("EOF")


