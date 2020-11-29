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
        veh.s1_mod = self.convert_state(veh)
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
        veh.s1_mod = self.convert_state(veh)
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
                # trainer.add_exp expects (states, actions, action_choices, rewards, critic)
                self.trainer.add_experience(tf.squeeze(veh.s0_mod),
                                            [action[0], action[1]],
                                            [veh.a0_choice[0], veh.a0_choice[1]],
                                            veh.reward, veh.critic)

        # Set past vehicle state and action pair
        veh.s0 = veh.s1
        veh.s0_mod = veh.s1_mod
        veh.a0 = veh.a1
        veh.a0_mod = veh.a1_mod
        veh.a0_choice = veh.a1_choice

        return veh.a1  # The hwsim library uses double precision floats

    def convert_state(self, veh):
        """
        Assembles state vector in TF form to pass to neural network.
        Normalises certain state variables and excludes constants.
        """

        # TODO Edit convert state method to remove unnecessary states
        # TODO Normalisation of input data? How to do gapB? Check with Bram

        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"]/(lane_width*3)  # Normalised
        max_vel = veh.s["maxVel"]  # Excluded
        curr_vel = veh.s["vel"][0]/max_vel  # Normalised and lateral component exluded

        # Current Lane:
        offset_current_lane_center = veh.s["laneC"]["off"]/(lane_width/2)  # Normalised
        rel_offset_back_center_lane = np.hstack((veh.s["laneC"]["relB"]["off"][0]/150,
                                                 veh.s["laneC"]["relB"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_back_center_lane = veh.s["laneC"]["relB"]["vel"]  # Normalised
        rel_offset_front_center_lane = np.hstack((veh.s["laneC"]["relF"]["off"][0]/150,
                                                  veh.s["laneC"]["relF"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_front_center_lane = veh.s["laneC"]["relF"]["vel"]/max_vel  # Normalised

        # Left Lane:
        offset_left_lane_center = veh.s["laneL"]["off"]/(lane_width/2)  # Normalised
        rel_offset_back_left_lane = np.hstack((veh.s["laneL"]["relB"]["off"][0]/150,
                                               veh.s["laneL"]["relB"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_back_left_lane = veh.s["laneL"]["relB"]["vel"]/max_vel  # Normalised
        rel_offset_front_left_lane = np.hstack((veh.s["laneL"]["relF"]["off"][0]/150,
                                                veh.s["laneL"]["relF"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_front_left_lane = veh.s["laneL"]["relF"]["vel"]/max_vel  # Normalised

        # Right Lane:
        offset_right_lane_center = veh.s["laneR"]["off"]/(lane_width/2)  # Normalised
        rel_offset_back_right_lane = np.hstack((veh.s["laneR"]["relB"]["off"][0]/150,
                                                veh.s["laneR"]["relB"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_back_right_late = veh.s["laneR"]["relB"]["vel"]/max_vel  # Normalised
        rel_offset_front_right_lane = np.hstack((veh.s["laneR"]["relB"]["off"][0]/150,
                                                 veh.s["laneR"]["relB"]["off"][1]/(lane_width/2)))  # Normalised to Dmax default
        rel_vel_front_right_late = veh.s["laneR"]["relB"]["vel"]/max_vel  # Normalised

        # Assemble state vector
        state = np.hstack((gap_to_road_edge, curr_vel,
                          offset_current_lane_center, rel_offset_back_center_lane, rel_vel_back_center_lane, rel_offset_front_center_lane, rel_vel_front_center_lane,
                          offset_left_lane_center, rel_offset_back_left_lane, rel_vel_back_left_lane, rel_offset_front_left_lane, rel_vel_front_left_lane,
                          offset_right_lane_center, rel_offset_back_right_lane, rel_vel_back_right_late, rel_offset_front_right_lane, rel_vel_front_right_late))

        state = tf.convert_to_tensor(state, dtype=tf.float32, name="state_input")  # 30 entries
        state = tf.expand_dims(state, 0)
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
            vel_controller = veh.s["vel"][0]-1
        elif vel_actions == 1:
            vel_controller = veh.s["vel"][0]
        elif vel_actions == 2:
            vel_controller = veh.s["vel"][0]+1
        else:
            print("Error with setting vehicle velocity!")
        v = tf.math.minimum(vel_controller, vel_bounds[1])
        sim_action = sim_action.write(0, tf.math.maximum(0, v))

        # TODO create logs to debug the offset not always obeying bounds!
        off_bounds = veh.a_bounds["off"]
        if off_actions == 0:  # Turn left
            off_controller = tf.math.maximum(off_bounds[0]+0.2, veh.s["laneC"]["off"]-0.05)
        elif off_actions == 1:  # Straight
            off_controller = tf.convert_to_tensor(veh.s["laneC"]["off"])
        elif off_actions == 2:  # Turn right
            off_controller = tf.math.minimum(off_bounds[1]-0.2, veh.s["laneC"]["off"]+0.05)
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
            r_s = 500*tf.math.exp(-(v_lim - v) ** 2 / 140) - 200*tf.math.exp(-(v) ** 2 / 70)

            # Collision??
            # TODO check collision punishment with Bram

            # Lane center reward:
            lane_offset = veh.s["laneC"]["off"]
            r_off = 200*tf.math.exp(-(lane_offset) ** 2 / 3.6)

            # Following distance:
            d_gap = 200*veh.s["laneC"]["relF"]["gap"][0]
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
