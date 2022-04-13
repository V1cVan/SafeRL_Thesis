import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # specify which GPU(s) to be used (1)
import pickle

from hwsim import Simulation, BasicPolicy, StepPolicy, SwayPolicy, IMPolicy, KBModel, TrackPolicy, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot


from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import pathlib
from Agents import *
from Policies import *
from HelperClasses import *
from Models import *

import multiprocessing as mp
import sys

# tf.config.experimental.set_visible_devices([], "GPU")
physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)


# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)
#

class Main(object):

    def __init__(self, n_vehicles, policy, model_param, training_param, tb_logger):
        # ...
        self.n_vehicles = n_vehicles
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
                self.simulate(self.training_param["sim_timesteps"], 1, 35)
                self.policy.agent.epsilon = self.policy.agent.prev_epsilon

            self.episodeTimer.startTime()
            # Set simulation environment

            if self.n_vehicles == "random":
                scenario = rand_sim_type(policy=self.policy, n_vehicles=self.n_vehicles)
            else:
                scenario = sim_type(policy=self.policy, n_vehicles=self.n_vehicles)
            self.sim = Simulation(scenario)

            with self.sim:
                # Loop through each timestep in episode.
                # Run the model for one episode to collect training data
                for t in np.arange(1, max_timesteps + 1):

                    # Perform one simulations step:
                    if not self.sim.stopped:
                        self.custom_action_timer.startTime()
                        self.sim.step()  # Calls AcPolicy.customAction method.
                        custom_action_time = self.custom_action_timer.endTime()

                        done = self.sim.stopped  # or self.sim._collision
                        if self.policy.agent.is_action_taken:
                            self.policy.agent.add_experience(done)
                            episode_reward_list.append(self.policy.agent.latest_experience[2])
                            curr_veh_speed = self.sim.vehicles[0].s["vel"][0] * 3.6
                            vehicle_speeds.append(curr_veh_speed)

                    if t % self.training_param["policy_rate"] == 0:
                        train_counter += 1
                        self.policy.agent.epsilon_decay_count = train_counter
                        self.policy.agent.timestep = int(t / self.training_param["policy_rate"])
                        if self.policy.agent.buffer.is_buffer_min_size():
                            model_update_counter += 1
                            if model_update_counter % self.training_param["model_update_rate"] == 0:
                                self.training_timer.startTime()
                                mean_batch_reward, loss, td_error, grads, clipped_grads = self.policy.agent.train_step()
                                epsilon = self.policy.agent.epsilon
                                train_time = self.training_timer.endTime()
                                episode_mean_batch_rewards.append(mean_batch_reward)
                                episode_losses.append(loss)
                                episode_td_errors.append(td_error)
                                trained = True
                                if self.training_param["use_per"]:
                                    beta = self.policy.agent.buffer.beta
                            if model_update_counter % self.training_param["target_update_rate"] == 0:
                                self.policy.agent.update_target_net()
                                # print("Updated target net.")

            reward = np.sum(episode_reward_list) / len(episode_reward_list)

            time_taken_episode = self.episodeTimer.endTime()
            # print(f"Time for episode = {time_taken_episode}")
            # Running reward smoothing effect
            running_reward = 0.05 * reward + (1 - 0.05) * running_reward
            if episode_count % 50 == 0 and trained:
                if self.training_param["use_per"]:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Beta = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon,
                                                         beta, time_taken_episode)
                else:
                    print_template = "Running reward = {:.3f} ({:.3f}) at episode {}. Loss = {:.3f}. Epsilon = {:.3f}. Episode timer = {:.3f}"
                    print_output = print_template.format(running_reward, reward, episode_count, loss, epsilon,
                                                         time_taken_episode)

                print(print_output)

            # Save episode training variables to tensorboard
            # TODO Variable to add logging of extended variables if needed
            # self.tb_logger.save_histogram("Episode mean batch rewards", x=episode_count, y=episode_mean_batch_rewards)
            # self.tb_logger.save_histogram("Episode losses", x=episode_count, y=episode_losses)
            # self.tb_logger.save_histogram("Episode TD errors", x=episode_count, y=episode_td_errors)
            # self.tb_logger.save_variable("Episode mean batch rewards (sum)", x=episode_count, y=np.sum(episode_mean_batch_rewards))
            # self.tb_logger.save_variable("Episode losses (sum)", x=episode_count, y=np.sum(episode_losses))
            # self.tb_logger.save_variable("Episode TD errors (sum)", x=episode_count, y=np.sum(episode_td_errors))
            # self.tb_logger.save_variable("Total episode reward (sum)", x=episode_count, y=np.sum(episode_reward_list))
            self.tb_logger.save_variable("Reward", x=episode_count,
                                         y=np.sum(episode_reward_list) / len(episode_reward_list))
            # self.tb_logger.save_variable("Total time taken for episode", x=episode_count, y=time_taken_episode)
            # self.tb_logger.save_variable("Total time taken for custom action", x=episode_count, y=custom_action_time)
            # self.tb_logger.save_variable("Total time taken for training iteration", x=episode_count, y=train_time)
            self.tb_logger.save_variable("Vehicle speed", x=episode_count, y=np.mean(vehicle_speeds))
            if (episode_count % 25 == 0 or episode_count == 1):
                if self.training_param["use_per"]:
                    try:
                        self.tb_logger.save_variable("Beta increment", x=episode_count, y=beta)
                    except:
                        self.tb_logger.save_variable("Beta increment", x=episode_count, y=self.policy.agent.buffer.beta)
                else:
                    try:
                        self.tb_logger.save_variable(name='Epsilon', x=episode_count, y=epsilon)
                    except:
                        self.tb_logger.save_variable(name='Epsilon', x=episode_count, y=self.policy.agent.epsilon)

            # Save model weights and biases and gradients of backprop.
            # self.tb_logger.save_weights_gradients(episode=episode_count,
            #                                  model=self.policy.agent.Q_actual_net,
            #                                  grads=grads,
            #                                  clipped_grads=clipped_grads)

            if running_reward >= self.training_param["final_return"] \
                    or episode_count == self.training_param["max_episodes"]:
                print_output = "Solved at episode {}!".format(episode_count)
                print(print_output)
                self.policy.agent.Q_actual_net.save_weights(self.model_param["weights_file_path"])
                break

            episode_count += 1

    def simulate(self, simulation_timesteps, simulation_episodes, total_vehicles):
        self.policy.agent.training = False
        self.policy.agent.evaluation = True

        episode = 0
        while episode < simulation_episodes:
            steps = 0
            print_counter = 0
            episode_reward_list = []
            vehicle_speeds = []
            mean_episode_reward = []
            mean_episode_speed = []
            with self.sim:
                self.create_plot()
                while not self.sim.stopped and steps < simulation_timesteps:  # and not self.p.closed:
                    self.sim.step()
                    episode_reward_list.append(self.policy.agent.latest_reward)
                    curr_veh_speed = self.sim.vehicles[0].s["vel"][0] * 3.6
                    vehicle_speeds.append(curr_veh_speed)
                    steps += 1
                    print_counter += 1
                    self.p.plot() # Uncomment for visualisation of run
                self.p.close() # Uncomment for visualisation of run
            episode += 1
            mean_episode_reward.append(np.mean(episode_reward_list))
            mean_episode_speed.append(np.mean(vehicle_speeds))


        mean_run_reward = np.mean(mean_episode_reward)
        mean_run_speed = np.mean(mean_episode_speed)
        self.tb_logger.save_variable("Average episode reward", x=total_vehicles, y=mean_run_reward)
        self.tb_logger.save_variable("Average episode speed", x=total_vehicles, y=mean_run_speed)
        self.policy.agent.evaluation = False
        self.policy.agent.training = True


def rand_sim_type(policy, n_vehicles):
    # Randomised highway
    # n_slow_veh = n_vehicles["slow"]
    # n_normal_veh = n_vehicles["medium"]
    # n_fast_veh = n_vehicles["fast"]
    # n_step = n_vehicles["step"]
    # n_sway = n_vehicles["sway"]
    # n_im = n_vehicles["im"]
    safetyCfg = {
        "Mvel": 1.0,
        "Gth": 2.0
    }
    sim_config = {
        "name": "Dense_Highway_Circuit",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": policy, "D_MAX": 160},
            {"amount": np.random.randint(1,9), "model": KBModel(), "policy": StepPolicy(np.random.randint(15,150),
                                                                        [np.random.randint(1,4)/10.0,
                                                                         np.random.randint(5,9)/10.0])},
            {"amount": np.random.randint(1,2), "model":  KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            # {"amount": n_im, "model": KBModel(), "policy": IMPolicy()},
            {"amount": np.random.randint(2,10), "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": np.random.randint(10,18), "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": np.random.randint(5,20), "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    }
    return sim_config

def sim_type(policy, n_vehicles):
    # Randomised highway
    n_slow_veh = n_vehicles["slow"]
    n_normal_veh = n_vehicles["medium"]
    n_fast_veh = n_vehicles["fast"]
    n_step = n_vehicles["step"]
    n_sway = n_vehicles["sway"]
    n_im = n_vehicles["im"]
    safetyCfg = {
        "Mvel": 1.0,
        "Gth": 2.0
    }
    sim_config = {
        "name": "Dense_Highway_Circuit",
        "scenario": "CIRCUIT",
        # "kM": 0,  # Max timesteps per episode enforced by simulator
        "k0": 0,
        "replay": False,
        "vehicles": [
            {"amount": 1, "model": KBModel(), "policy": policy, "D_MAX": 160},
            # {"amount": n_step, "mode  KBModel(), "policy": SwayPolicy(), "N_OV": 2, "safety": safetyCfg},
            # {"amount": n_im, "model": KBModel(), "policy": IMPolicy()},
            {"amount": n_slow_veh, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": n_normal_veh, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": n_fast_veh, "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    }
    return sim_config


def start_run(run_type, vehicles, method, parameter, seed, value):
    # Start training loop using given arguments here..
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")
    config.scenarios_path = str(SC_PATH)
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm")

    if value == "DDQN":
        USE_LSTM = False
        USE_DEEPSET = False
    elif value == "LSTM_Deepset":
        USE_LSTM = True
        USE_DEEPSET = True
    elif value == "LSTM":
        USE_LSTM = True
        USE_DEEPSET = False
    elif value == "Deepset":
        USE_LSTM = False
        USE_DEEPSET = True
    else:
        print("value not set properly")
        sys.exit()

    """RUN PARAMETERS:"""
    SEED = seed
    RUN_TYPE = run_type  # "train"/test
    RUN_NAME = method
    # For sweeps
    # if "elu" in str(value):
    #     if "relu" in str(value):
    #         if "relu6" in str(value):
    #             RUN_INFO = parameter + "=" + "Relu6"
    #         elif "leaky_relu" in str(value):
    #             RUN_INFO = parameter + '=' + "Leaky Relu"
    #         else:
    #             RUN_INFO = parameter + "=" + "Relu"
    #     else:
    #         RUN_INFO = parameter + "=" + "Elu"
    # elif "tanh" in str(value):
    #     RUN_INFO = parameter + "=" + "Tanh"
    # else:
    #     RUN_INFO = parameter + "=" + str(value)

    # For method comparison
    RUN_INFO = parameter + "=" + str(value)

    # total_vehicles = vehicles["slow"] + vehicles["medium"] + vehicles["fast"]
    total_vehicles = 0
    if RUN_TYPE == "train":
        SAVE_DIRECTORY = "logfiles/" + RUN_NAME + "/" + RUN_TYPE + "/Seed" + str(
            SEED) + "-Details=" + str(RUN_INFO)
    else:
        SAVE_DIRECTORY = "logfiles/" + RUN_NAME + "/" + RUN_TYPE + "/Seed" + str(SEED) + "-n_veh=" + str(
            total_vehicles) + "-Details=" + str(RUN_INFO)

    run_settings = {
        "run_type": RUN_TYPE,
        "run_name": RUN_NAME,
        "run_info": RUN_INFO,
        "save_directory": SAVE_DIRECTORY,
        "n_vehicles": vehicles
    }

    config.seed = SEED
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    """ BASE MODEL PARAMETERS:"""
    # Base parameters (DDQN):
    N_UNITS = (32, 32)
    REMOVE_STATE_VELOCITY = False
    if REMOVE_STATE_VELOCITY == True:
        N_INPUTS = 31
    else:
        N_INPUTS = 55
    N_ACTIONS = 5
    ACT_FUNC = tf.nn.relu

    BATCH_NORM = False
    """ NON-TEMPORAL MODEL PARAMETERS: """
    # For deepset:
    PHI_SIZE = (32, 64)
    RHO_SIZE = (64, 32)
    ACT_FUNC_PHI = tf.nn.relu
    ACT_FUNC_RHO = tf.nn.relu
    # For CNN's:
    FILTERS = (15, 15)  # Dimensionality of output space
    KERNEL = 4  # Convolution width
    STRIDES = (1, 1)  # Stride size
    USE_POOLING = False
    TEMPORAL_CNN_TYPE = '2D'  # 3D or 2D
    NORMAL_CNN_TYPE = 1
    # 0=1D conv. on vehicle dim.,
    # 1=1D conv. on measurements dim-With pooling over vehicle dimension then renders it permutation invariant
    # 2=2D conv. on vehicle and measurements dimensions,
    # For LSTM:
    LSTM_UNITS = 32



    if RUN_TYPE == "test":
        MODEL_FILE_PATH = "logfiles/" + RUN_NAME + "/" + "train" + "/Seed" + str(500) + "-Details=" + str(
            RUN_INFO) + "/"
        TRAINABLE = False
    elif RUN_TYPE == "train":
        MODEL_FILE_PATH = "logfiles/" + RUN_NAME + "/" + "train" + "/Seed" + str(SEED) + "-Details=" + str(
            RUN_INFO) + "/"
        TRAINABLE = True
    else:
        print("Run type not correctly specified!")
        sys.exit()

    model_param = {
        "n_units": N_UNITS,
        "n_inputs": N_INPUTS,
        "n_actions": N_ACTIONS,
        "activation_function": ACT_FUNC,
        "weights_file_path": MODEL_FILE_PATH,
        "seed": SEED,
        "trainable": TRAINABLE,  # For batch normalisation to freeze layers
        "batch_normalisation": BATCH_NORM,
        'remove_velocity': REMOVE_STATE_VELOCITY,
        "deepset_param": {

            "n_units_phi": PHI_SIZE,
            "act_func_phi": ACT_FUNC_PHI,
            "n_units_rho": RHO_SIZE,
            "act_func_rho": ACT_FUNC_RHO
        },
        "cnn_param": {
            'kernel': KERNEL,
            'filters': FILTERS,
            'strides': STRIDES,
            'use_pooling': USE_POOLING,
            # 0=1D conv. on vehicle dim.,
            # 1=1D conv. on measurements dim.,
            # 2=2D conv. on vehicle and measurements dimensions,
            'normal_CNN_type': NORMAL_CNN_TYPE,
            'temporal_CNN_type': TEMPORAL_CNN_TYPE,  # 2D or 3D
        },
        "LSTM_param": {
            'n_units': LSTM_UNITS,
        }
    }

    """TRAINING PARAMETERS:"""
    POLICY_ACTION_RATE = 8  # Number of simulator steps before new control action is taken
    MAX_TIMESTEPS = 5e3  # range: 5e3 - 10e3
    MAX_EPISODES = 800
    FINAL_RETURN = 1
    SHOW_TRAIN_PLOTS = False
    SAVE_TRAINING = True
    LOG_FREQ = 0
    PLOT_FREQ = 15
    SIM_TIMESTEPS = 5e3
    if RUN_TYPE == 'test':
        SIM_EPISODES = 50
    else:
        SIM_EPISODES = 1
    BUFFER_SIZE = 300000
    BATCH_SIZE = 32  # range: 32 - 150
    EPSILON_MIN = 1  # Exploration
    EPSILON_MAX = 0.1  # Exploitation
    DECAY_RATE = 0.999976
    MODEL_UPDATE_RATE = 1
    TARGET_UPDATE_RATE = 1e4
    LEARN_RATE = 0.0001  # range: 1e-3 - 1e-4
    OPTIMISER = tf.optimizers.Adam(learning_rate=LEARN_RATE)
    LOSS_FUNC = tf.losses.Huber()  # tf.losses.Huber()  # PER loss function is MSE
    GAMMA = 0.98 # range: 0.9 - 0.99
    CLIP_GRADIENTS = False
    CLIP_NORM = 2
    # Reward weights = (rew_vel, rew_lat_lane_position, rew_fol_dist, staying_right, collision penalty)
    REWARD_WEIGHTS = np.array([1.25, 0.15, 0.8, 0.35, 0.0])
    STANDARDISE_RETURNS = False
    USE_PER = False
    ALPHA = 0.75  # Priority scale: a=0:random, a=1:completely based on priority
    BETA = 0.25  # Prioritisation factor
    BETA_INCREMENT = 2e-6  # Rate of Beta annealing to 1
    # Model types:
    USE_TARGET_NETWORK = True
    USE_DUELLING = False
    # USE_DEEPSET = False
    USE_CNN = False
    USE_TEMPORAL_CNN = False
    # USE_LSTM = False
    # REMOVE_STATE_VELOCITY  -- MOVED UP
    ADD_NOISE = False

    # RewardFunction().plot_reward_functions()
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
        "use_temporal_CNN": USE_TEMPORAL_CNN,
        "use_LSTM": USE_LSTM,
        "noise_param": {"use_noise": ADD_NOISE, "magnitude": 0.05, "normal": True, "mu": 0.0, "sigma": 0.707,
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

    # Initialise model type:
    if USE_DEEPSET:
        DQ_net = DeepSetQNetwork(model_param=model_param)
        DQ_target_net = DeepSetQNetwork(model_param=model_param)
    elif USE_CNN :
        DQ_net = CNN(model_param=model_param)
        DQ_target_net = CNN(model_param=model_param)
    elif USE_TEMPORAL_CNN:
        DQ_net = TemporalCNN(model_param=model_param)
        DQ_target_net = TemporalCNN(model_param=model_param)
    elif USE_DUELLING:
        DQ_net = DuellingDqnNetwork(model_param=model_param)
        DQ_target_net = DuellingDqnNetwork(model_param=model_param)
    elif USE_LSTM:
        DQ_net = LSTM(model_param=model_param)
        DQ_target_net = LSTM(model_param=model_param)
    elif USE_LSTM and USE_DEEPSET:
        DQ_net = LstmDeepSetNetwork(model_param=model_param)
        DQ_target_net = LstmDeepSetNetwork(model_param=model_param)
        DQ_net.display_overview()
    else:
        # Final checks: To avoid mistakes in parallel runs
        assert not (USE_TEMPORAL_CNN and USE_LSTM)
        assert not (USE_DEEPSET and USE_CNN)
        assert not (USE_DEEPSET and USE_TEMPORAL_CNN)
        assert not (USE_CNN and USE_TEMPORAL_CNN)
        assert not (USE_DUELLING)

        DQ_net = DeepQNetwork(model_param=model_param)
        DQ_target_net = DeepQNetwork(model_param=model_param)

    # Initialise agent and policy:
    if USE_TARGET_NETWORK == True:
        dqn_agent = DqnAgent(network=DQ_net,
                             target_network=DQ_target_net,
                             training_param=training_param,
                             tb_logger=tb_logger)
    else:
        dqn_agent = DqnAgent(network=DQ_net,
                             target_network=DQ_net,
                             training_param=training_param,
                             tb_logger=tb_logger)

    dqn_policy = DiscreteSingleActionPolicy(agent=dqn_agent)



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
        # main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
        # TODO Tidy up simulation part:
        # Simulate model:
        main.policy.agent.Q_actual_net.load_weights(MODEL_FILE_PATH)
        main.policy.agent.evaluation = True
        main.simulate(simulation_timesteps=SIM_TIMESTEPS, simulation_episodes=SIM_EPISODES,
                      total_vehicles=total_vehicles)
    else:
       main.train_policy()

    print(f"Number Veh={vehicles}; Method={method}; Parameter={parameter}; Value={value}")


if __name__ == "__main__":

    run_timer = Timer("Run timer")
    run_timer.startTime()

    PROCS = 1  # Number of cores to use
    mp.set_start_method("spawn")  # Make sure different workers have different seeds if applicable
    P = mp.cpu_count()  # Number of available cores
    procs = max(min(PROCS, P), 1)  # Clip number of procs to [1;P]

    # with tf.device('/CPU:0'):
    def param_gen():
        """
        This function yields all the parameter combinations to try...
        arguments:
            vehicles = defintion of the vehicles in the simulation
            method = method name
            parameter = parameter name
            seed = seed for run
            value = parameter value
        """
        # vehicles = {"slow": 10, "medium": 20, "fast": 5}  # total = 35 // default

        veh_0 = {"slow": 2, "medium": 5, "fast": 1}  # total = 8
        veh_1 = {"slow": 3, "medium": 8, "fast": 2}  # total = 13
        veh_2 = {"slow": 4, "medium": 10, "fast": 2}  # total = 16
        veh_3 = {"slow": 6, "medium": 13, "fast": 3}  # total = 22
        veh_4 = {"slow": 8, "medium": 16, "fast": 4}  # total = 28
        veh_5 = {"slow": 10, "medium": 20, "fast": 5, "im": 0, "step": 0, "sway": 0}  # total = 35
        veh_6 = {"slow": 12, "medium": 24, "fast": 6}  # total = 42
        veh_7 = {"slow": 15, "medium": 28, "fast": 7}  # total = 50
        veh_8 = {"slow": 20, "medium": 35, "fast": 10}  # total = 65
        veh_9 = {"slow": 25, "medium": 55, "fast": 15}  # total = 95
        veh_10 = {"slow": 35, "medium": 70, "fast": 25}  # total = 130

        # veh_many_fast = {"slow": 10, "medium": 10, "fast": 20, "im": 0, "step": 0, "sway": 0}  # total = 40
        # veh_extended = {"slow": 8, "medium": 18, "fast": 8, "im": 2, "step": 3, "sway": 1}  # total = 40
        # veh_stress = {"slow": 10, "medium": 15, "fast": 17, "im": 4, "step": 6, "sway": 2}  # total = 54

        run_type = "train"
        method = "LSTM_Deepset"
        vehicles = veh_5
        parameter = "Model"
        for seed in (100,  300, 500):
            for value in ("LSTM_Deepset", "LSTM", "Deepset", "DDQN"):
                yield run_type, vehicles, method, parameter, seed, value

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

    print(f"Total run time = {run_timer.endTime() / 60} minutes.")
    print("EOF")