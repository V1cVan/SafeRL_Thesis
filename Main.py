from hwsim import Simulation, BasicPolicy, KBModel, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot

import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from implementation/NeuralModels import *
from implementation/RL_Policies import *

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
    def createPlot(self,sim):
        shape = (4, 2)
        groups = [([0, 2], 0)]
        vehicle_type = "car" if FANCY_CARS else "cuboid3D"
        self.p = Plotter(sim, "Multi car simulation", mode=PLOT_MODE, shape=shape, groups=groups, off_screen=OFF_SCREEN)
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


    def trainPolicy(self):
        action_history = []
        critic_history = []
        rewards_history = []
        running_reward = 0
        episode_count = 0
        policy = self.pol[0]["policy"]

        # Run until all episodes are completed (reward reached).
        while True:
            # Set simulation environment
            with self.sim:
                self.createPlot(self.sim)
                vehicle = self.sim.vehicles[0]
                policy.init_vehicle(vehicle)  ## call sim will do the same
                episode_reward = 0
                timestep = 0
                # Loop through each timestep in episode.
                while not self.sim.stopped:

                    # Perform one simulations step:
                    self.sim.step()  # Calls AcPolicy.customAction method.
                    # ... (visualization/extra callbacks)
                    self.p.plot()
                    if timestep > 0:
                        episode_reward += policy.trainer.buffer[-1][2]

                    if timestep == 100:
                        print("break")
                    timestep += 1
                else:
                    # TODO look at adding max_timesteps_per_episode to ensure you learn even when the policy is already pretty good
                    # Or update policy throughout
                    running_reward = 0.05 * episode_reward + (1-0.05)*running_reward
                    # Perform one train step, using the collected new experience:
                    policy.trainer.trainStep()

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

    # Model configuration and settings
    model_param = {
        "gamma": 0.99,  # Discount factor
        "n_nodes": [400, 200],  # Number of hidden nodes in each layer
        "n_layers": 2,  # Number of layers
        "n_inputs": 47,  # Standard size of S
        "n_actions": 2
    }
    training_param = {
        "max_steps_per_episode": 10000,  # TODO kM - max value of k
        "final_return": 150,
        "optimiser": keras.optimizers.Adam(learning_rate=0.02),
        "loss_function": keras.losses.Huber()
    }

    # Initialise network/model architecture:
    actor_net = ActorNet(model_param)
    actor_net.displayOverview()
    critic_net = CriticNet(model_param)
    critic_net.displayOverview()
    trainer = GradAscentTrainer(actor_net, critic_net, training_param)  # training method used

    # Simulation configuration and settings
    veh_types = [
        {"amount": 1, "model": KBModel(), "policy": AcPolicy(trainer)},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("slow")},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("normal")},
        {"amount": 20, "model": KBModel(), "policy": BasicPolicy("fast")}
    ]
    sim_config = {
        "name": "AC_policy",
        "scenario": "CIRCUIT",
        "kM": 1000,  # TODO  for max_timesteps_per_episode
        "k0": 0,
        "replay": False,
        "vehicles": veh_types
    }

    sim = Simulation(sim_config)

    # Set up main class for running simulations:
    main = Main(sim_config)

    # Train model:
    main.trainPolicy()

    # Simulate model:


    print("EOF")


