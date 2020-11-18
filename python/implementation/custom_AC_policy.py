import pathlib
import random
import numpy as np
from hwsim import Simulation, BasicPolicy, StepPolicy, KBModel, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import time
import timeit





class FixedLanePolicy(CustomPolicy, enc_name="fixed_lane"):
    """ Simple policy where each vehicle will stay in its initial lane with a certain target
    velocity (relative to the maximum allowed speed). The actual velocity is always upper
    bounded by the safety bounds (taking vehicles in front into account)."""

    def __init__(self):
        super().__init__()
        self.STEP_TIME = 100 # Change reference velocity every 100 iterations (10s)

    def init_vehicle(self, veh):
        """ Policy objects are shared over many different vehicles so to associate
        attributes to specific vehicles, we can use this method (which is called
        during Vehicle instantiation) """
        veh.rel_vel = 0
        veh.counter = 0

    def _set_rel_vel(self, veh):
        veh.rel_vel = 0.95-random.random()*0.3

    def custom_action(self, veh):
        """ This method is called at every iteration and the returned numpy arrary
        will be used as the new reference actions (passed to the lower level controllers
        who will set up proper model inputs to track the new reference) """
        # Start with updating the counter and setting a new reference if necessary
        veh.counter -= 1
        if veh.counter<=0:
            veh.counter = self.STEP_TIME
            self._set_rel_vel(veh)
        # Then calculate proper actions from the current reference
        s = veh.s  # Current augmented state
        bounds = veh.a_bounds # Current safety bounds on the actions (calculated from the current augmented state). Vehicle operation remains 'safe' as long as we respect these bounds.
        v_max = veh.rel_vel*(s["maxVel"])
        v = min(v_max,bounds["vel"][1])
        v = max(0,v)
        # Final actions are: the target velocity and negating the offset towards the lane center
        return np.array([v,-s["laneC"]["off"]])

class AC_policy(CustomPolicy, enc_name="customAC"):
    """ Simple policy where each vehicle will stay in its initial lane with a certain target
    velocity (relative to the maximum allowed speed). The actual velocity is always upper
    bounded by the safety bounds (taking vehicles in front into account)."""

    def __init__(self):
        super().__init__()
        self.STEP_TIME = 1 # Change reference velocity every 100 iterations (10s)
        # init model
        #self.model = model

    def init_vehicle(self, veh):
        """ Policy objects are shared over many different vehicles so to associate
        attributes to specific vehicles, we can use this method (which is called
        during Vehicle instantiation) """
        veh.rel_vel = 0
        veh.counter = 0

    def _set_rel_vel(self, veh):
        veh.rel_vel = 0.95-random.random()*0.3

    def custom_action(self, veh):
        """ This method is called at every iteration and the returned numpy arrary
        will be used as the new reference actions (passed to the lower level controllers
        who will set up proper model inputs to track the new reference) """

        # action, critic_prob = self.model(state)

        # Start with updating the counter and setting a new reference if necessary
        veh.counter -= 1
        if veh.counter<=0:
            veh.counter = self.STEP_TIME
            self._set_rel_vel(veh)
        # Then calculate proper actions from the current reference
        s = veh.s  # Current augmented state
        bounds = veh.a_bounds # Current safety bounds on the actions (calculated from the current augmented state). Vehicle operation remains 'safe' as long as we respect these bounds.
        v_max = veh.rel_vel*(s["maxVel"])
        v = min(v_max,bounds["vel"][1])
        v = max(0,v)
        # Final actions are: the target velocity and negating the offset towards the lane center
        return np.array([v,-s["laneC"]["off"]])

def createModel(model_param):
    """createModel(model_param): Creates keras model of the neural network
    *kwargs:
        n_nodes = [layer1, layer 2, ...]
        n_layers = #
        n_inputs = #
        n_actions = #
    """
    # Add variability in depth
    inputs = layers.Input(shape=(model_param["n_inputs"],))
    # def createLayer(prev_layer,index):
    #     next_layer = layers.Dense(model_param["n_nodes"][index], activation="relu")(prev_layer)
    #     return next_layer
    layer1 = layers.Dense(model_param["n_nodes"][0], activation="relu")(inputs)
    layer2 = layers.Dense(model_param["n_nodes"][1], activation="relu")(layer1)
    actor = layers.Dense(model_param["n_actions"], activation="softmax")(layer2)
    critic = layers.Dense(1)(layer2)

    model = keras.Model(inputs=inputs, outputs=[actor, critic])
    model.summary()  # Display overview of model
    return model


def trainModel(model, training_param):
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    action_collision_history = []
    action_safeact_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    p = training_param
    # Run until all episodes completed (reward level reached)
    with sim:
        veh = sim.vehicles[0]  # Is this the ego vehicle?
        # Set environment with random state
        state = veh.s_raw
        episode_reward = 0
        with tf.GradientTape() as tape:
            # Run through each timestep in episode
            for timestep in range(1, p["max_steps_per_episode"]):
                # Render incremental plotting if desired
                if timestep == 1:
                    createPlot()
                state = veh.s_raw
                state = tf.convert_to_tensor(state)  # shape = (47,)
                state = tf.expand_dims(state,0)  # shape = (1,47)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action, critic_prob = model(state)
                critic_value_history.append(critic_prob[0, 0])


                print()


                # action_probs_history.append(tf.math.log(action_probs[0, action]))


                # Apply the sampled action in our environment
                # state, reward, done, info = env.step(action)
                # We also have to define rewards
                # rewards_history.append(reward)
                # episode_reward += reward

                # break for loop if done with episode.

            # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            # Return = SUM_t=0^inf (gamma*reward_t)
            #returns = []
            #discounted_sum = 0
            # for r in rewards_history[::-1]:
            #     discounted_sum = r + gamma * discounted_sum
            #     returns.insert(0, discounted_sum)

            # Normalize
            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()

            # Calculating loss values of each timestep in episode to update our network for next episode
            # history = zip(action_probs_history, critic_value_history, returns)
            # actor_losses = []
            # critic_losses = []
            # for actor_log_prob, crit_value, ret in history:
            #     # At this point in history, the critic estimated that we would get a
            #     # total reward = `value` in the future. We took an action with log probability
            #     # of `log_prob` and ended up receiving a total reward = `ret`.
            #     # The actor must be updated so that it predicts an action that leads to
            #     # high rewards (compared to critic's estimate) with high probability.
            #     diff = ret - crit_value
            #     actor_losses.append(-actor_log_prob * diff)  # actor loss
            #
            #     # The critic must be updated so that it predicts a better estimate of
            #     # the future rewards.
            #     critic_losses.append(
            #         loss_function(tf.expand_dims(crit_value, 0), tf.expand_dims(ret, 0))
            #     )

            # # Backpropagation
            # loss_value = sum(actor_losses) + sum(critic_losses)
            # grads = tape.gradient(loss_value, model.trainable_variables)
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # # Clear the loss and reward history
            # action_probs_history.clear()
            # critic_value_history.clear()
            # rewards_history.clear()

        # # Log details
        # episode_count += 1
        # if episode_count % 10 == 0:
        #     template = "running reward: {:.2f} at episode {}"
        #     print(template.format(running_reward, episode_count))
        #
        # if running_reward >= final_return:  # Condition to consider the task solved
        #     print("Solved at episode {}!".format(episode_count))
        #     break

    #return the model

    return ""


def runSimulation(simulated_timesteps):
    # initstate = initialised environment
    # Loop through timesteps to simulate everything
        # render environment
        # measure current state

        # action to take based on model:
        # action_probabilities, critic_values = model(state)
        # action = np.random.choice(num_actions, p=np.squeeze(action_probabilities))

        # get observation from environment based on action taken:
        # observation = env.step(action)
        # set new state: state = observation

        # check if simulation done and close plots and simulation
    return ""

def createPlot():
    shape = (4,2)
    groups = [(np.s_[:],0)]
    plot = Plotter(sim,"Fixed lane simulation",mode=Plotter.Mode.LIVE,shape=shape,groups=groups)
    plot.subplot(0,0)
    plot.add_text("Detail view")
    DetailPlot(plot,show_ids=True)
    plot.add_overlay()
    SimulationPlot(plot,vehicle_type=None,show_marker=True)
    plot.subplot(0,1)
    plot.add_text("Front view")
    BirdsEyePlot(plot,vehicle_type="car",view=BirdsEyePlot.View.FRONT)
    plot.subplot(1,1)
    plot.add_text("Actions")
    lines = {
        "rel_vel": {
            "color": [0, 0, 0],
            "getValue": lambda vehicle: vehicle.rel_vel
        }
    }
    TimeChartPlot(plot, lines, None, "rel_vel", [0])
    plot.subplot(2,1)
    ActionsPlot(plot,actions="vel")
    plot.subplot(3,1)
    ActionsPlot(plot,actions="off")
    plot.plot()


if __name__=="__main__":
    # Set configuration parameters for the whole setup
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    config.seed = 1000 # For reproducability
    print(f"Using seed {config.seed}")

    input_dir = ""
    LOG_DIR = "logs"
    output_dir = LOG_DIR
    kbm = KBModel()
    custom_policy = AC_policy()
    fixed_lane_policy = FixedLanePolicy()
    veh_settings = {"model": kbm, "policy": custom_policy, "R": 0, "l": 3.6, "s": 0}
    vTypes = [
        ## Adding my vehicle like this gives no attribute rel_vel in plotting ??
        # {"model": kbm, "policy": custom_policy, "R": 0, "l": 0, "s": 0, "v": 50},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("slow")},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy("normal")},
        {"amount": 20, "model": KBModel(), "policy": BasicPolicy("fast")}
    ]
    sConfig = {
        "name": "AC_policy",
        "scenario": "CIRCUIT",
        "kM": 1000,
        "k0": 0,
        "replay": True,
        "vehicles": vTypes
    }
    sim = Simulation(sConfig)
    sim.add_vehicles([
        {"model": kbm, "policy": custom_policy, "R": 0, "l": 0, "s": 0, "v": 50},
        {"model": kbm, "policy": fixed_lane_policy, "R": 0, "l": 0, "s": 10, "v": 120},
        {"model": kbm, "policy": fixed_lane_policy, "R": 0, "l": 3.6, "s": 20, "v": 30},
        {"model": kbm, "policy": fixed_lane_policy, "R": 0, "l": -3.6, "s": 1, "v": 70}
    ])

    model_param = {
        "gamma": 0.99,  # Discount factor
        "n_nodes": [400, 200],  # Number of hidden nodes in each layer
        "n_layers": 2  # Number of layers
    }
    training_param = {
        "max_steps_per_episode": 10000,
        "final_return": 150,
        "optimiser": keras.optimizers.Adam(learning_rate=0.02),
        "loss_function": keras.losses.Huber()
    }

    with sim:
        veh = sim.vehicles[0]
        model_param["n_inputs"] = np.size(veh.s_raw)
        model_param["n_actions"] = np.size(veh.u_raw)

    # Create NN model
    myModel = createModel(model_param)
    # Train Model
    trained_model = trainModel(myModel, training_param)



    # Train model
    # trainedModel = trainModel(kwargs)
    # save weights
    # option to load model

    # set Timesteps to simulate
    # simulate trained model



    simulate(sim)






##### some old code #####
# def simulate(sim):
#     # Runs the given simulation and plots it
#     with sim:
#         shape = (2, 2)
#         shape = (4, 2)
#         groups = [(np.s_[:],0)]
#         vehicle_type = "car" if FANCY_CARS else "box"
#         p = Plotter(sim, "Multi car simulation", mode=Plotter.Mode.LIVE, shape=shape, groups=groups, off_screen=OFF_SCREEN)
#         p.V = 64
#         p.subplot(0, 0)
#         p.add_text("Detail view")
#         DetailPlot(p, show_ids=True)
#         p.add_overlay()
#         SimulationPlot(p, vehicle_type=None, show_marker=True)
#         p.subplot(0, 1)
#         p.add_text("Front view")
#         BirdsEyePlot(p, vehicle_type=vehicle_type, view=BirdsEyePlot.View.FRONT)
#         p.subplot(1, 1)
#         p.add_text("Rear view")
#         BirdsEyePlot(p, vehicle_type=vehicle_type, view=BirdsEyePlot.View.REAR)
#         p.subplot(2,1)
#         p.add_text("Actions")
#         ActionsPlot(p,actions="vel")  # plot velocity actions
#         p.subplot(3,1)
#         ActionsPlot(p,actions="off")  # Plot offset
#         p.subplot(2,1)
#         p.subplot(3,1)
#         p.plot()  # Initial plot
#
#         while not sim.stopped and not p.closed:
#             # start = timeit.default_timer()
#             sim.step()
#             # end = timeit.default_timer()
#             # print(f"Simulation step took {(end - start) * 1000} ms")
#             # start = timeit.default_timer()
#             p.plot()
#             # end = timeit.default_timer()
#             # print(f"Drawing took {(end - start) * 1000} ms")
#
#         p.close()  # Make sure everything is closed correctly (e.g. video file is closed properly)
#
#
# if __name__=="__main__":
#     ID = -1  # ID of simulation to replay or -1 to create a new one
#     PLOT_MODE = Plotter.Mode.LIVE
#     OFF_SCREEN = False
#     FANCY_CARS = True
#     LOG_DIR = "logs"
#     ROOT = pathlib.Path(__file__).resolve().parents[2]
#     SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")
#
#     config.scenarios_path = str(SC_PATH)
#     # config.seed = 1249517370
#     print(f"Using seed {config.seed}")
#     if ID < 0:
#         ID = int(time.time())
#         print(f"Creating new multi car simulation with ID {ID}")
#         input_dir = ""
#         output_dir = LOG_DIR
#         vTypes = [
#             {"amount": 20, "model": KBModel(), "policy": BasicPolicy("slow")},
#             {"amount": 40, "model": KBModel(), "policy": BasicPolicy("normal")},
#             {"amount": 10, "model": KBModel(), "policy": BasicPolicy("fast")}
#         ]
#     else:
#         print(f"Replaying multi car simulation with ID {ID}")
#         input_dir = LOG_DIR
#         output_dir = ""
#         vTypes = []
#
#     sConfig = {
#         "name": f"multi_car_{ID}",
#         "scenario": "CIRCUIT",
#         "kM": 1000,
#         "input_dir": input_dir,
#         "k0": 0,
#         "replay": True,
#         "output_dir": output_dir,
#         "vehicles": vTypes
#     }
#
#     sim = Simulation(sConfig)
#     simulate(sim)