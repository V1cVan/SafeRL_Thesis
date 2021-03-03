from hwsim import CustomPolicy, ActionType
import tensorflow as tf
import numpy as np
import random

class DiscreteActionPolicy(CustomPolicy):
    """
    DDQN!!!!
    """
    LONG_ACTION = ActionType.REL_VEL
    LAT_ACTION = ActionType.LANE

    def __init__(self, trainer):
        super(DiscreteActionPolicy, self).__init__()
        self.trainer = trainer  # trainer = f(NN_model)
        self.STEP_TIME = self.trainer.training_param["STEP_TIME"]


    def init_vehicle(self, veh):
        # Book-keeping of last states and actions
        # s0, a0, c0 = previous vehicle state and action-critic pair
        # s1, a1, c0 = current vehicle state action-critic pair
        veh.counter = 0
        veh.s0 = None
        veh.s0_mod = None
        veh.s1 = None
        veh.s1_mod = None
        veh.a0 = None
        veh.a0_mod = None
        veh.a0_choice = None
        veh.prev_action = None
        veh.rew_buffer = []
        veh.c0 = None
        veh.a1_mod = None
        veh.a1_choice = None
        veh.a1 = None
        veh.c1 = None
        veh.flag = None


    def custom_action(self, veh):
        # s0, a0 = previous vehicle state action pair
        # s1, a1 = current vehicle state action pair
        veh.counter -= 1

        if veh.counter <= 0:
            veh.counter = self.STEP_TIME
            # Set current vehicle state and action pair
            veh.s1 = veh.s_raw
            veh.s1_mod = self.convert_state(veh)
            Q = self.get_action(veh.s1_mod)
            self.trainer.states.append(veh.s1_mod)
            self.trainer.actions.append(Q)
            veh.a1_mod = Q
            action_choice = self.trainer.get_action_choice(Q)
            veh.a1_choice = action_choice
            veh.a1 = self.convert_action_discrete(veh, action_choice)


            # Save experience
            if veh.a0_mod is not None:  # Check if the agent has taken an action that led to this reward...
                # Calculate reward at current state (if action was taken previously)
                veh.flag = self.trainer.stop_flags
                veh.reward = self.get_reward(veh) + np.sum(veh.rew_buffer)
                self.trainer.rewards.append(veh.reward)
                if self.trainer is not None and self.trainer.training is True:
                    # Save action taken previously on previous state value
                    # action = velocity action, lane_change action
                    action = veh.a0_mod[0][veh.a0_choice]

                    # Save to buffer from the Buffer class in HelperClasses.py module
                    # add_experience expects (timestep, state, vel_model_action, off_model_action,
                    #                         vel_action_sim, offset_action_sim, vel_choice, off_choice, reward, critic)
                    self.trainer.buffer.set_experience(state=np.squeeze(veh.s0_mod),
                                                       action=veh.a0_choice,
                                                       reward=veh.reward,
                                                       next_state=np.squeeze(veh.s1_mod),
                                                       done_flag=veh.flag)

            # Set past vehicle state and action pair
            veh.s0 = veh.s1
            veh.s0_mod = veh.s1_mod
            veh.a0 = veh.a1
            veh.a0_mod = veh.a1_mod
            veh.a0_choice = veh.a1_choice
            veh.c0 = veh.c1
            veh.rew_buffer = []
            output = np.array([veh.a1[0], veh.a1[1]], dtype=np.float64)  # The hwsim library uses double precision floats
            veh.prev_action = action_choice
            return output

        else:
            discrete_actions = self.convert_action_discrete(veh, veh.prev_action)
            output = np.array(discrete_actions, dtype=np.float64)
            veh.rew_buffer.append(self.get_reward(veh))
            return output


    def convert_state(self, veh):
        """
        Assembles state vector in TF form to pass to neural network.
        Normalises certain state variables and excludes constants.
        """

        # Normalise states and remove unnecessary states:
        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"]/(lane_width*3)  # Normalised
        max_vel = veh.s["maxVel"]           # Excluded
        curr_vel = veh.s["vel"][0]/max_vel  # Normalised longitudinal component and lateral component excluded


        # Current Lane:
        offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
        rel_offset_back_center_lane = np.hstack((veh.s["laneC"]["relB"]["gap"][:, 0] / 150,
                                                 veh.s["laneC"]["relB"]["gap"][:, 1] / lane_width))  # Normalised to Dmax default
        rel_vel_back_center_lane = np.matrix.flatten(veh.s["laneC"]["relB"]["vel"]) / max_vel  # Normalised

        rel_offset_front_center_lane = np.hstack((veh.s["laneC"]["relF"]["gap"][:, 0] / 150,
                                                  veh.s["laneC"]["relF"]["gap"][:, 1] / lane_width))  # Normalised to Dmax default
        rel_vel_front_center_lane = np.matrix.flatten(veh.s["laneC"]["relF"]["vel"]) / max_vel  # Normalised

        # Left Lane:
        offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
        rel_offset_back_left_lane = np.hstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / 150,
                                               veh.s["laneL"]["relB"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
        rel_offset_back_left_lane = np.squeeze(rel_offset_back_left_lane)
        rel_vel_back_left_lane = np.matrix.flatten(veh.s["laneL"]["relB"]["vel"]) / max_vel  # Normalised

        rel_offset_front_left_lane = np.hstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / 150,
                                                veh.s["laneL"]["relF"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
        rel_offset_front_left_lane = np.squeeze(rel_offset_front_left_lane)
        rel_vel_front_left_lane = np.matrix.flatten(veh.s["laneL"]["relF"]["vel"]) / max_vel  # Normalised

        # Right Lane:
        offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised
        rel_offset_back_right_lane = np.hstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / 150,
                                                veh.s["laneR"]["relB"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
        rel_offset_back_right_lane = np.squeeze(rel_offset_back_right_lane)
        rel_vel_back_right_late = np.matrix.flatten(veh.s["laneR"]["relB"]["vel"]) / max_vel  # Normalised

        rel_offset_front_right_lane = np.hstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / 150,
                                                 veh.s["laneR"]["relF"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
        rel_offset_front_right_lane = np.squeeze(rel_offset_front_right_lane)
        rel_vel_front_right_late = np.matrix.flatten(veh.s["laneR"]["relF"]["vel"]) / max_vel  # Normalised

        # Assemble state vector
        state = np.hstack((gap_to_road_edge, curr_vel,
                          offset_current_lane_center, rel_offset_back_center_lane, rel_vel_back_center_lane, rel_offset_front_center_lane, rel_vel_front_center_lane,
                          offset_left_lane_center, rel_offset_back_left_lane, rel_vel_back_left_lane, rel_offset_front_left_lane, rel_vel_front_left_lane,
                          offset_right_lane_center, rel_offset_back_right_lane, rel_vel_back_right_late, rel_offset_front_right_lane, rel_vel_front_right_late))

        state = tf.convert_to_tensor(state, dtype=tf.float32, name="state_input")
        state = tf.expand_dims(state, 0)
        return state  # Can be overridden by subclasses

    def get_action(self, state):
        """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
        # Receive action and critic values from NN:
        Q = self.trainer.Q_actual_net(state)
        return Q

    def convert_action_discrete(self, veh, action_choices):
        """
        Get the action vector that will be passed to the vehicle from the given model action vector
        (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a
        4 discrete actions: slow, maintain ,acc, left, centre, right
        """
        sim_action = np.array([0, 0], dtype=np.float32)

        # TODO add penalty for selecting improper actions

        # Safety bounds
        vel_bounds = veh.a_bounds["long"]  # [min_rel_vel, max_rel_vel]
        off_bounds = veh.a_bounds["lat"]

        # Compute safe actions
        if action_choices == 0:  # Slow down, stay in lane
            vel_controller = np.maximum(vel_bounds[0], -5)
            steer_controller = 0
        elif action_choices == 1:  # Constant speed, stay in lane
            vel_controller = np.minimum(vel_bounds[1], 0)
            steer_controller = 0
        elif action_choices == 2:  # Speed up, stay in lane
            vel_controller = np.minimum(vel_bounds[1], +5)
            steer_controller = 0
        elif action_choices == 3:  # Constant speed, turn left
            vel_controller = np.minimum(vel_bounds[1], 0)
            steer_controller = np.maximum(off_bounds[0], -1)
        elif action_choices == 4:  # Constant speed, turn right
            vel_controller = np.minimum(vel_bounds[1], 0)
            steer_controller = np.minimum(off_bounds[1], 1)
        else:
            print("Error with setting offset action!")

        sim_action[0] = vel_controller
        sim_action[1] = steer_controller
        return sim_action

    def get_reward(self, veh=None):
        """
        Calculate reward for actions.
        Reward = Speed + LaneCentre + FollowingDistance
        """
        # Reward function weightings:
        w_vel = self.trainer.reward_weights[0]  # Speed weight
        w_off = self.trainer.reward_weights[1]  # Lane center
        w_dist = self.trainer.reward_weights[2]  # Following distance
        w_stay_right = self.trainer.reward_weights[3] # Staying in right lane

        # Reward function declaration:
        reward = np.array([0], dtype=np.float32)
        if veh is not None:
            # Velocity reward:
            v = np.squeeze(veh.s["vel"])[0]
            v_lim = 120 / 3.6
            r_vel = np.exp(-(v_lim - v) ** 2 / 140) #- np.exp(-(v) ** 2 / 70)

            # # TODO remove lane centre reward when acting with discrete lane changes
            # # Lane center reward:
            lane_offset = np.squeeze(veh.s["laneC"]["off"])
            r_off = np.exp(-(lane_offset) ** 2 / 3.6)

            # Following distance:
            d_gap = veh.s["laneC"]["relF"]["gap"][0, 0]
            d_lim = 0
            r_follow = -np.exp(-(d_lim - d_gap) ** 2 / 100)

            # Stay right:
            r_right = (1/(veh.s["laneC"]["width"]*3))*veh.s["gapB"][0]

            reward = w_vel*r_vel + w_off*r_off + w_dist*r_follow + w_stay_right*r_right

        else:
            reward = 0
        return reward

    # def squeeze_vehicle_state(self, veh):
    #     # Squeeze values to fix matrix inconsistencies
    #     # Current lane:
    #     veh.s["laneC"]["off"] = np.squeeze(veh.s["laneC"]["off"])
    #     veh.s["laneC"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
    #     veh.s["laneC"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
    #     veh.s["laneC"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
    #     veh.s["laneC"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
    #     # Left lane
    #     veh.s["laneL"]["off"] = np.squeeze(veh.s["laneC"]["off"])
    #     veh.s["laneL"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
    #     veh.s["laneL"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
    #     veh.s["laneL"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
    #     veh.s["laneL"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
    #     # Right lane
    #     veh.s["laneR"]["off"] = np.squeeze(veh.s["laneC"]["off"])
    #     veh.s["laneR"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
    #     veh.s["laneR"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
    #     veh.s["laneR"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
    #     veh.s["laneR"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
    #
    #     return veh.s



# class DiscreteActionPolicy(CustomPolicy):
#     """
#     Actor-critic on-policy RL controller for highway decision making.
#     """
#     LONG_ACTION = ActionType.REL_VEL
#     LAT_ACTION = ActionType.LANE
#
#     def __init__(self, trainer):
#         super(DiscreteActionPolicy, self).__init__()
#         self.trainer = trainer  # trainer = f(NN_model)
#         self.STEP_TIME = self.trainer.training_param["STEP_TIME"]
#
#
#     def init_vehicle(self, veh):
#         # Book-keeping of last states and actions
#         # s0, a0, c0 = previous vehicle state and action-critic pair
#         # s1, a1, c0 = current vehicle state action-critic pair
#         veh.counter = 0
#         veh.s0 = None
#         veh.s0_mod = None
#         veh.s1 = None
#         veh.s1_mod = None
#         veh.a0 = None
#         veh.a0_mod = None
#         veh.a0_choice = None
#         veh.prev_action = None
#         veh.rew_buffer = []
#         veh.c0 = None
#         veh.a1_mod = None
#         veh.a1_choice = None
#         veh.a1 = None
#         veh.c1 = None
#
#     def custom_action(self, veh):
#         # s0, a0 = previous vehicle state action pair
#         # s1, a1 = current vehicle state action pair
#         veh.counter -= 1
#
#         if veh.counter <= 0:
#             veh.counter = self.STEP_TIME
#             # Set current vehicle state and action pair
#             veh.s1 = veh.s_raw
#             veh.s1_mod = self.convert_state(veh)
#             action_probs, veh.c1 = self.get_action_and_critic(veh.s1_mod)
#             self.trainer.states.append(veh.s1_mod)
#             self.trainer.actions.append(action_probs)
#             veh.a1_mod = action_probs
#             action_choice = self.trainer.get_action_choice(action_probs)
#             veh.a1_choice = action_choice
#             veh.a1 = self.convert_action_discrete(veh, action_choice)
#
#             # Save experience
#             if veh.a0_mod is not None:  # Check if the agent has taken an action that led to this reward...
#                 # Calculate reward at current state (if action was taken previously)
#                 veh.reward = self.get_reward(veh) + np.sum(veh.rew_buffer)
#                 self.trainer.rewards.append(veh.reward)
#                 if self.trainer is not None and self.trainer.training is True:
#                     # Save action taken previously on previous state value
#                     # action = velocity action, lane_change action
#                     action = veh.a0_mod[0][veh.a0_choice]
#
#                     # Save to buffer from the Buffer class in HelperClasses.py module
#                     # add_experience expects (timestep, state, vel_model_action, off_model_action,
#                     #                         vel_action_sim, offset_action_sim, vel_choice, off_choice, reward, critic)
#                     self.trainer.buffer.set_experience(timestep=self.trainer.timestep,
#                                                        state=np.squeeze(veh.s0_mod),
#                                                        action_model=np.array(action),
#                                                        action_sim=veh.a0,
#                                                        action_choice=veh.a0_choice,
#                                                        reward=veh.reward,
#                                                        critic=np.squeeze(veh.c0))
#
#             # Set past vehicle state and action pair
#             veh.s0 = veh.s1
#             veh.s0_mod = veh.s1_mod
#             veh.a0 = veh.a1
#             veh.a0_mod = veh.a1_mod
#             veh.a0_choice = veh.a1_choice
#             veh.c0 = veh.c1
#             veh.rew_buffer = []
#             output = np.array([veh.a1[0], veh.a1[1]], dtype=np.float64)  # The hwsim library uses double precision floats
#             veh.prev_action = action_choice
#             return output
#
#         else:
#             discrete_actions = self.convert_action_discrete(veh, veh.prev_action)
#             output = np.array(discrete_actions, dtype=np.float64)
#             veh.rew_buffer.append(self.get_reward(veh))
#             return output
#
#
#     def convert_state(self, veh):
#         """
#         Assembles state vector in TF form to pass to neural network.
#         Normalises certain state variables and excludes constants.
#         """
#
#         # Normalise states and remove unnecessary states:
#         lane_width = veh.s["laneC"]["width"]  # Excluded
#         gap_to_road_edge = veh.s["gapB"]/(lane_width*3)  # Normalised
#         max_vel = veh.s["maxVel"]           # Excluded
#         curr_vel = veh.s["vel"][0]/max_vel  # Normalised longitudinal component and lateral component excluded
#
#
#         # Current Lane:
#         offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
#         rel_offset_back_center_lane = np.hstack((veh.s["laneC"]["relB"]["gap"][:, 0] / 150,
#                                                  veh.s["laneC"]["relB"]["gap"][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_vel_back_center_lane = np.matrix.flatten(veh.s["laneC"]["relB"]["vel"]) / max_vel  # Normalised
#
#         rel_offset_front_center_lane = np.hstack((veh.s["laneC"]["relF"]["gap"][:, 0] / 150,
#                                                   veh.s["laneC"]["relF"]["gap"][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_vel_front_center_lane = np.matrix.flatten(veh.s["laneC"]["relF"]["vel"]) / max_vel  # Normalised
#
#         # Left Lane:
#         offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
#         rel_offset_back_left_lane = np.hstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / 150,
#                                                veh.s["laneL"]["relB"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_offset_back_left_lane = np.squeeze(rel_offset_back_left_lane)
#         rel_vel_back_left_lane = np.matrix.flatten(veh.s["laneL"]["relB"]["vel"]) / max_vel  # Normalised
#
#         rel_offset_front_left_lane = np.hstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / 150,
#                                                 veh.s["laneL"]["relF"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_offset_front_left_lane = np.squeeze(rel_offset_front_left_lane)
#         rel_vel_front_left_lane = np.matrix.flatten(veh.s["laneL"]["relF"]["vel"]) / max_vel  # Normalised
#
#         # Right Lane:
#         offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised
#         rel_offset_back_right_lane = np.hstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / 150,
#                                                 veh.s["laneR"]["relB"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_offset_back_right_lane = np.squeeze(rel_offset_back_right_lane)
#         rel_vel_back_right_late = np.matrix.flatten(veh.s["laneR"]["relB"]["vel"]) / max_vel  # Normalised
#
#         rel_offset_front_right_lane = np.hstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / 150,
#                                                  veh.s["laneR"]["relF"]["gap"][0][:, 1] / lane_width))  # Normalised to Dmax default
#         rel_offset_front_right_lane = np.squeeze(rel_offset_front_right_lane)
#         rel_vel_front_right_late = np.matrix.flatten(veh.s["laneR"]["relF"]["vel"]) / max_vel  # Normalised
#
#         # Assemble state vector
#         state = np.hstack((gap_to_road_edge, curr_vel,
#                           offset_current_lane_center, rel_offset_back_center_lane, rel_vel_back_center_lane, rel_offset_front_center_lane, rel_vel_front_center_lane,
#                           offset_left_lane_center, rel_offset_back_left_lane, rel_vel_back_left_lane, rel_offset_front_left_lane, rel_vel_front_left_lane,
#                           offset_right_lane_center, rel_offset_back_right_lane, rel_vel_back_right_late, rel_offset_front_right_lane, rel_vel_front_right_late))
#
#         state = tf.convert_to_tensor(state, dtype=tf.float32, name="state_input")
#         state = tf.expand_dims(state, 0)
#         return state  # Can be overridden by subclasses
#
#     def get_action_and_critic(self, state):
#         """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
#         # Receive action and critic values from NN:
#         action_probs, critic = self.trainer.actor_critic_net(state)
#         return action_probs, critic
#
#     def convert_action_discrete(self, veh, action_choices):
#         """
#         Get the action vector that will be passed to the vehicle from the given model action vector
#         (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a
#         4 discrete actions: slow, maintain ,acc, left, centre, right
#         """
#         sim_action = np.array([0, 0], dtype=np.float32)
#
#         # TODO add penalty for selecting improper actions
#
#         # Safety bounds
#         vel_bounds = veh.a_bounds["long"]  # [min_rel_vel, max_rel_vel]
#         off_bounds = veh.a_bounds["lat"]
#
#         # Compute safe actions
#         if action_choices == 0:  # Slow down, stay in lane
#             vel_controller = np.maximum(vel_bounds[0], -5)
#             steer_controller = 0
#         elif action_choices == 1:  # Constant speed, stay in lane
#             vel_controller = np.minimum(vel_bounds[1], 0)
#             steer_controller = 0
#         elif action_choices == 2:  # Speed up, stay in lane
#             vel_controller = np.minimum(vel_bounds[1], +5)
#             steer_controller = 0
#         elif action_choices == 3:  # Constant speed, turn left
#             vel_controller = np.minimum(vel_bounds[1], 0)
#             steer_controller = np.maximum(off_bounds[0], -1)
#         elif action_choices == 4:  # Constant speed, turn right
#             vel_controller = np.minimum(vel_bounds[1], 0)
#             steer_controller = np.minimum(off_bounds[1], 1)
#         else:
#             print("Error with setting offset action!")
#
#         sim_action[0] = vel_controller
#         sim_action[1] = steer_controller
#         return sim_action
#
#     def get_reward(self, veh=None):
#         """
#         Calculate reward for actions.
#         Reward = Speed + LaneCentre + FollowingDistance
#         """
#         # Reward function weightings:
#         w_vel = self.trainer.reward_weights[0]  # Speed weight
#         w_off = self.trainer.reward_weights[1]  # Lane center
#         w_dist = self.trainer.reward_weights[2]  # Following distance
#         w_stay_right = self.trainer.reward_weights[3] # Staying in right lane
#
#         # Reward function declaration:
#         reward = np.array([0], dtype=np.float32)
#         if veh is not None:
#             # Velocity reward:
#             v = np.squeeze(veh.s["vel"])[0]
#             v_lim = 120 / 3.6
#             r_vel = np.exp(-(v_lim - v) ** 2 / 140) #- np.exp(-(v) ** 2 / 70)
#
#             # # TODO remove lane centre reward when acting with discrete lane changes
#             # # Lane center reward:
#             lane_offset = np.squeeze(veh.s["laneC"]["off"])
#             r_off = np.exp(-(lane_offset) ** 2 / 3.6)
#
#             # Following distance:
#             d_gap = veh.s["laneC"]["relF"]["gap"][0, 0]
#             d_lim = 0
#             r_follow = -np.exp(-(d_lim - d_gap) ** 2 / 100)
#
#             # Stay right:
#             r_right = (1/(veh.s["laneC"]["width"]*3))*veh.s["gapB"][0]
#
#             reward = w_vel*r_vel + w_off*r_off + w_dist*r_follow + w_stay_right*r_right
#
#         else:
#             reward = 0
#         return reward
#
#     # def squeeze_vehicle_state(self, veh):
#     #     # Squeeze values to fix matrix inconsistencies
#     #     # Current lane:
#     #     veh.s["laneC"]["off"] = np.squeeze(veh.s["laneC"]["off"])
#     #     veh.s["laneC"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
#     #     veh.s["laneC"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
#     #     veh.s["laneC"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
#     #     veh.s["laneC"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
#     #     # Left lane
#     #     veh.s["laneL"]["off"] = np.squeeze(veh.s["laneC"]["off"])
#     #     veh.s["laneL"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
#     #     veh.s["laneL"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
#     #     veh.s["laneL"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
#     #     veh.s["laneL"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
#     #     # Right lane
#     #     veh.s["laneR"]["off"] = np.squeeze(veh.s["laneC"]["off"])
#     #     veh.s["laneR"]["relB"]["off"] = np.squeeze(veh.s["laneC"]["relB"]["off"])
#     #     veh.s["laneR"]["relB"]["vel"] = np.squeeze(veh.s["laneC"]["relB"]["vel"])
#     #     veh.s["laneR"]["relF"]["off"] = np.squeeze(veh.s["laneC"]["relF"]["off"])
#     #     veh.s["laneR"]["relF"]["vel"] = np.squeeze(veh.s["laneC"]["relF"]["vel"])
#     #
#     #     return veh.s


class FixedLanePolicy(CustomPolicy, enc_name="fixed_lane"):
    """ Simple policy where each vehicle will stay in its initial lane with a certain target
    velocity (relative to the maximum allowed speed). The actual velocity is always upper
    bounded by the safety bounds (taking vehicles in front into account)."""
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.REL_OFF # Alternatively: ActionType.LANE

    def __init__(self, speed):
        super().__init__()
        self.STEP_TIME = 100 # Change reference velocity every 100 iterations (10s)
        self.speed = speed

    def init_vehicle(self, veh):
        """ Policy objects are shared over many different vehicles so to associate
        attributes to specific vehicles, we can use this method (which is called
        during Vehicle instantiation) """
        veh.abs_vel = self.speed
        veh.counter = 0

    def _set_rel_vel(self, veh):
        veh.rel_vel = 0

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
        s = veh.s # Current augmented state
        bounds = veh.a_bounds # Current safety bounds on the actions (calculated from the current augmented state). Vehicle operation remains 'safe' as long as we respect these bounds.
        v_max = veh.rel_vel*(s["maxVel"])
        v = min(v_max,bounds["long"][1])
        v = max(0,v)
        # Final actions are: the target velocity and negating the offset towards the lane center
        return np.array([self.speed,-s["laneC"]["off"]])
        # Alternatively (with LANE actionType):
        # return np.array([v,0]) # Lane reference is 0 => remain in (center of) current lane
