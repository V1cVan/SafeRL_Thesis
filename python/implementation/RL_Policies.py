from hwsim import CustomPolicy
import tensorflow as tf
import logging
import numpy as np

class AcPolicyDiscrete(CustomPolicy):
    """
    Actor-critic on-policy RL controller for highway decision making.
    """

    def __init__(self, trainer):
        super(AcPolicyDiscrete, self).__init__()
        self.trainer = trainer  # trainer = f(NN_model)
        logging.basicConfig(level=logging.INFO, filename="./python/implementation/logfiles/ACPolicyDiscrete.log")
        with open('./python/implementation/logfiles/ACPolicyDiscrete.log', 'w'):
            pass  # Clear the log file of previous run

    def init_vehicle(self, veh):
        # Book-keeping of last states and actions
        # s0, a0 = previous vehicle state action pair
        # s1, a1 = current vehicle state action pair

        veh.s0 = None
        veh.s0_mod = None
        veh.s1 = veh.s_raw
        veh.s1_mod = self.convert_state(veh.s_raw)
        veh.a0 = None
        veh.a0_mod = None
        veh.a0_choice = None
        veh.a1_mod = None
        veh.a1_choice = None

        veh.a1 = None

    def custom_action(self, veh):
        # s0, a0 = previous vehicle state action pair
        # s1, a1 = current vehicle state action pair

        # Set current vehicle state and action pair
        veh.s1 = veh.s_raw
        veh.s1_mod = self.convert_state(veh.s1)
        action_vel_probs, action_off_probs, veh.critic = self.get_action_and_critic(veh.s1_mod)
        veh.a1_mod = [action_vel_probs, action_off_probs]
        action_choice_vel, action_choice_off = self.trainer.get_action_choice([action_vel_probs, action_off_probs])
        veh.a1_choice = [action_choice_vel, action_choice_off]
        veh.a1 = self.convert_action_discrete(veh, [action_choice_vel, action_choice_off])

        # Save experience
        if veh.a0_mod is not None:
            # Calculate reward at current state (if action was taken previously)
            veh.reward = self.get_reward(veh)
            if self.trainer is not None:
                # Save action taken previously on previous state value
                action = veh.a0_mod[0][0, veh.a0_choice[0]], veh.a0_mod[1][0, veh.a0_choice[1]]
                self.trainer.add_experience(tf.get_static_value(tf.squeeze(veh.s0_mod)), action,
                                            [veh.a0_choice[0], veh.a0_choice[1]], veh.reward, tf.squeeze(veh.critic))

        # Set past vehicle state and action pair
        veh.s0 = veh.s1
        veh.s0_mod = veh.s1_mod
        veh.a0 = veh.a1
        veh.a0_mod = veh.a1_mod
        veh.a0_choice = veh.a1_choice

        return veh.a1  # The hwsim library uses double precision floats

    def convert_state(self, state):
        """ Get the modified state vector that will be passed to the actor and critic models from the
        state vector (available in veh). I.e. the mapping s->s_mod """
        # TODO Edit convert state method to remove unnecessary states
        state = tf.convert_to_tensor(state, dtype=tf.float32, name="state_input")  # shape = (47,)
        state = tf.expand_dims(state, 0)  # shape = (1,47)
        return state  # Can be overridden by subclasses

    def get_action_and_critic(self, state):
        """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
        action_vel_probs, action_off_probs, critic_prob = self.trainer.actor_critic_net(state)
        return action_vel_probs, action_off_probs, critic_prob

    def convert_action_discrete(self, veh, action_choices):
        """
        Get the action vector that will be passed to the vehicle from the given model action vector
        (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a
        4 discrete actions: slow, maintain ,acc, left, centre, right
        """
        sim_action = tf.TensorArray(size=0, dtype=tf.float64, dynamic_size=True)
        vel_actions, off_actions = action_choices

        vel_bounds = veh.a_bounds["vel"]
        if vel_actions == 0:
            vel_controller = veh.s["vel"][0]-10
        elif vel_actions == 1:
            vel_controller = veh.s["vel"][0]
        elif vel_actions == 2:
            vel_controller = veh.s["vel"][0]+10
        else:
            print("Error with setting vehicle velocity!")
        v = tf.math.minimum(vel_controller, vel_bounds[1])
        sim_action = sim_action.write(0, tf.math.maximum(0, v))

        # TODO create logs to debug the offset not always obeying bounds!
        off_bounds = veh.a_bounds["off"]
        if off_actions == 0:  # Turn left
            off_controller = tf.math.maximum(off_bounds[0]+0.2, veh.s["laneC"]["off"]-0.5)
        elif off_actions == 1:  # Straight
            off_controller = tf.convert_to_tensor(veh.s["laneC"]["off"])
        elif off_actions == 2:  # Turn right
            off_controller = tf.math.minimum(off_bounds[1]-0.2, veh.s["laneC"]["off"]+0.5)
        else:
            print("Error with setting offset action!")
        sim_action = sim_action.write(1, off_controller)

        sim_action = sim_action.stack()
        return tf.get_static_value(sim_action)  # output is array

    def get_reward(self, veh=None):
        """
        Calculate reward for actions.
        Reward = Speed + LaneCentre + FollowingDistance
        """
        reward = tf.Variable(0, dtype=tf.float64)
        if veh is not None:
            # Velocity reward:
            v = veh.s["vel"][0]
            v_lim = 120 / 3.6
            r_s = 1*tf.math.exp(-(v_lim - v) ** 2 / 140)

            # Collision??
            # TODO check collision punishment with Bram

            # Lane center reward:
            lane_offset = veh.s["laneC"]["off"]
            r_off = tf.math.exp(-(lane_offset) ** 2 / 3.6)

            # Following distance:
            d_gap = veh.s["laneC"]["relF"]["gap"][0]
            d_lim = 10
            r_follow = -tf.math.exp(-(d_lim - d_gap) ** 2 / 20)

            reward.assign(r_s + r_off + r_follow)
        else:
            reward.assign(0.0)
        return tf.dtypes.cast(reward, tf.float32)

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
