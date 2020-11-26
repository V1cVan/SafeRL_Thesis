from hwsim import Simulation, BasicPolicy, StepPolicy, KBModel, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AcPolicyDiscrete(CustomPolicy):
    """
    Actor-critic on-policy RL controller for highway decision making.
    """

    def __init__(self, trainer):
        super(AcPolicyDiscrete, self).__init__()
        self.trainer = trainer  # trainer = f(NN_model)

    def init_vehicle(self, veh):
        # Bookkeeping of last states and actions
        veh.s0 = None  # Previous vehicle state
        veh.s0_mod = None  # Previous vehicle state as passed to the actor and critic models
        veh.s1 = veh.s_raw  # Current vehicle state
        veh.s1_mod = self.convertState(veh.s_raw)  # Current vehicle state as passed to the actor and critic models
        veh.a0 = None
        veh.a0_mod = None
        veh.a1_mod = self.getAction(veh.s1_mod)
        veh.a1 = self.convertActionDiscrete(veh, veh.a1_mod)
        veh.reward = self.getReward()

    def custom_action(self, veh):
        # s0, a0 = previous vehicle state action pair
        # s1, a1 = current vehicle state action pair
        veh.s0 = veh.s1
        veh.s0_mod = veh.s1_mod
        veh.a0 = veh.a1
        veh.a0_mod = veh.a1_mod
        veh.s1 = veh.s_raw
        veh.s1_mod = self.convertState(veh.s1)
        veh.a1_mod = self.getAction(veh.s1_mod)
        veh.a1 = self.convertActionDiscrete(veh, veh.a1_mod)
        veh.critic = self.getCritic(veh.s1_mod)

        # Calculate new metrics
        veh.reward = self.getReward(veh)
        # And report new experience to trainer, if available
        if self.trainer is not None:
            vel_action, off_action = self.getActionChoice(veh.a1_mod)
            action = [tf.math.log(veh.a1_mod[0, vel_action]), tf.math.log(veh.a1_mod[1, off_action])]
            self.trainer.addExperience(veh.s0_mod, action, veh.reward, veh.s1_mod, veh.critic[0, 0])

        return veh.a1  # The hwsim library uses double precision floats

    def convertState(self, state):
        """ Get the modified state vector that will be passed to the actor and critic models from the
        state vector (available in veh). I.e. the mapping s->s_mod """
        # TODO Edit convert state method to remove unnecessary states
        state = state.astype(np.float32)
        state = tf.convert_to_tensor(state)  # shape = (47,)
        state = tf.expand_dims(state, 0)  # shape = (1,47)

        return state  # Can be overridden by subclasses

    def getAction(self, state):
        """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
        action_probs = self.trainer.actor_net(state)
        action_probs = tf.convert_to_tensor(np.squeeze(action_probs))
        return action_probs

    def getActionChoice(self, action_prob):
        """ Randomly choose from the available actions."""
        vel_actions_choice = np.random.choice(np.size(action_prob[0]), p=np.squeeze(action_prob[0]))
        off_actions_choice = np.random.choice(np.size(action_prob[1]), p=np.squeeze(action_prob[1]))
        return vel_actions_choice, off_actions_choice

    def convertActionDiscrete(self, veh, action):
        """
        Get the action vector that will be passed to the vehicle from the given model action vector
        (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a
        4 discrete actions: slow, maintain ,acc, left, centre, right
        """
        sim_action = np.zeros([1, 2])
        vel_actions, off_actions = self.getActionChoice(action)

        vel_bounds = veh.a_bounds["vel"]
        if vel_actions == 0:
            vel_controller = veh.s["vel"][0]-1
        elif vel_actions == 1:
            vel_controller = veh.s["vel"][0]
        elif vel_actions == 2:
            vel_controller = veh.s["vel"][0]+1
        else:
            print("Error with setting vehicle velocity!")
        v = min(vel_controller, vel_bounds[1])
        sim_action[0, 0] = max(0, v)

        off_bounds = veh.a_bounds["off"]
        if off_actions == 0:  # Turn left
            off_controller = max(off_bounds[0], veh.s["laneC"]["off"]-3.5)
        elif off_actions == 1:  # Straight
            off_controller = veh.s["laneC"]["off"]
        elif off_actions == 2:  # Turn right
            off_controller = min(off_bounds[1], veh.s["laneC"]["off"]+3.5)
        else:
            print("Error with setting offset action!")
        sim_action[0, 1] = off_controller

        return np.squeeze(sim_action).view(np.float64)  # output is array

    def getCritic(self, state):
        """ Get the critic value for the current state transition"""
        return self.trainer.critic_net(state)

    def getReward(self, veh=None):
        """
        Calculate reward for actions.
        Reward = Speed + LaneCentre + FollowingDistance
        """
        if veh is not None:
            # Velocity reward:
            v = veh.s["vel"][0]
            v_lim = 120 / 3.6
            r_s = np.exp(-(v - v_lim) ** 2 / 10)

            # Collision??

            # Lane center reward:
            lane_offset = veh.s["laneC"]["off"]
            r_off = np.exp(-(lane_offset) ** 2 / 1)

            # Following distance:
            # TODO just take long. component
            d_gap = np.linalg.norm(veh.s["laneC"]["relF"]["gap"])
            d_lim = 10
            r_follow = np.exp(-(d_gap - d_lim) ** 2 / 2)

            reward = r_s + r_off + r_follow
        else:
            reward = 0
        return reward

        # def convertActionContinuous(self, veh):
        #     """ Get the action vector that will be passed to the vehicle from the given model action vector
        #     (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a """
        #     if veh.a1 is not None:
        #         veh.a1 = veh.a1.__array__().astype(np.float32)
        #
        #         v_max = veh.s["maxVel"]  # 30m/s
        #         vel_med = v_max / 2.0
        #         vel_bounds = veh.a_bounds["vel"]
        #         vel_controller = vel_med + veh.a1_mod[0, 0] * vel_med
        #         v = min(vel_controller, vel_bounds[1])
        #         veh.a1[0, 0] = max(0, v)
        #
        #
        #         off_bounds = veh.a_bounds["off"]
        #         off_controller = 3.6 * veh.a1_mod[0, 1]
        #         if off_controller <= 0:
        #             off = max(off_bounds[0], off_controller)
        #         elif 0 < off_controller:
        #             off = min(off_bounds[1], off_controller)
        #         else:
        #             # TODO Debug
        #             print("Offset action bound error")
        #             off = veh.s["laneC"]["off"]
        #         veh.a1[0, 1] = off
        #
        #         veh.a1 = tf.convert_to_tensor(veh.a1)
        #         return veh.a1  # Can be overridden by subclasses
        #     else:
        #         return veh.a1_mod