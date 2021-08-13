from hwsim import CustomPolicy, ActionType
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import deque


def add_state_noise(state, is_normal=False, is_uniform=False, magnitude=0.0, mu=0.0, sigma=0.0):
    """
    Adds sensor simulated sensor noise to state measurements to make the learning setting more difficult.
        mu = mean for gaussian noise
        sigma = std dev. for gaussian noise
        magnitude = magnitude of gaussian and uniform noise
    """

    show_noise_plot = False
    if show_noise_plot:
        plt.figure()
        plt.subplot(211)
        normal_noise = np.random.normal(mu, sigma, size=1000)
        plt.hist(normal_noise, bins=50, density=True)
        plt.title("Normal Noise")
        plt.grid(True)
        plt.subplot(212)
        low, high = -0.1, 0.1
        uniform_noise = np.random.uniform(low, high, size=1000)
        plt.hist(uniform_noise, bins=50, density=True)
        plt.title("Uniform Noise")
        plt.grid(True)
        plt.show()

    noisy_state = state
    size_state = len(state)

    if size_state == 1:  # Singular state matrix
        shape = tf.shape(state)
        normal_noise = magnitude * np.random.normal(mu, sigma, size=shape)
        uniform_noise = magnitude * np.random.uniform(low=-0.3, high=0.3, size=shape)

    else:  # State matrix has static and dynamic components
        # Add noise to the dynamic part of the state matrix
        state_0 = state[0]
        shape_0 = tf.shape(state_0)
        normal_noise_0 = magnitude * np.random.normal(mu, sigma, size=shape_0)
        uniform_noise_0 = magnitude * np.random.uniform(low=-0.3, high=0.3, size=shape_0)

        # Add noise to the static part of the state matrix
        state_1 = state[1]
        shape_1 = tf.shape(state_1)
        normal_noise_1 = magnitude * np.random.normal(mu, sigma, size=shape_1)
        uniform_noise_1 = magnitude * np.random.uniform(low=-0.3, high=0.3, size=shape_1)

    if is_normal and is_uniform:
        if size_state == 1:  # Singular state matrix
            noisy_state = state + normal_noise + uniform_noise
        else:  # State matrix has static and dynamic components
            state_0 = state_0 + normal_noise_0 + uniform_noise_0
            state_1 = state_1 + normal_noise_1 + uniform_noise_1
            noisy_state = [state_0, state_1]
    elif is_normal and not is_uniform:
        if size_state == 1:  # Singular state matrix
            noisy_state = state + normal_noise
        else:  # State matrix has static and dynamic components
            state_0 = state_0 + normal_noise_0
            state_1 = state_1 + normal_noise_1
            noisy_state = [state_0, state_1]
    elif not is_normal and is_uniform:
        if size_state == 1:  # Singular state matrix
            noisy_state = state + uniform_noise
        else:  # State matrix has static and dynamic components
            state_0 = state_0 + uniform_noise_0
            state_1 = state_1 + uniform_noise_1
            noisy_state = [state_0, state_1]
    else:
        print("No noise to be added to state!")

    return noisy_state

def convert_state(veh, remove_velocity):
    """
    Assembles state vector in TF form to pass to neural network.
    Normalises certain state variables and excludes constants.
    """
    D_MAX = veh.D_MAX

    if remove_velocity == False:
        D_MAX = 150

        # Static part of the state vector:
        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"] / (lane_width * 3)  # Normalised
        max_vel = veh.s["maxVel"]  # Excluded
        curr_vel = veh.s["vel"] / max_vel  # Normalised longitudinal component and lateral component excluded

        offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
        offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
        offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised

        static_state = np.hstack(
            (gap_to_road_edge, curr_vel, offset_current_lane_center, offset_left_lane_center, offset_right_lane_center))
        static_state = tf.expand_dims(tf.convert_to_tensor(static_state, dtype=tf.float32, name="static_state_input"),
                                      0)

        # Dynamic part of the state vector:
        # Current lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relB"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_back_center_lane = np.vstack((veh.s["laneC"]["relB"]["gap"][:, 0] / D_MAX,
                                                 veh.s["laneC"]["relB"]["gap"][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_center_lane = veh.s["laneC"]["relB"]["vel"] / max_vel  # Normalised
        vehicles_back_center_lane = vehicles * np.hstack((rel_offset_back_center_lane, rel_vel_back_center_lane))
        dynamic_state = vehicles_back_center_lane

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relF"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_front_center_lane = np.vstack((veh.s["laneC"]["relF"]["gap"][:, 0] / D_MAX,
                                                  veh.s["laneC"]["relF"]["gap"][:,
                                                  1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_center_lane = veh.s["laneC"]["relF"]["vel"] / max_vel  # Normalised
        vehicles_front_center_lane = vehicles * np.hstack((rel_offset_front_center_lane, rel_vel_front_center_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_center_lane))

        # Left lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_left_lane = np.vstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / D_MAX,
                                               veh.s["laneL"]["relB"]["gap"][0][:,
                                               1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_left_lane = veh.s["laneL"]["relB"]["vel"][0] / max_vel  # Normalised
        vehicles_back_left_lane = vehicles * np.hstack((rel_offset_back_left_lane, rel_vel_back_left_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_back_left_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_left_lane = np.vstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneL"]["relF"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_left_lane = veh.s["laneL"]["relF"]["vel"][0] / max_vel  # Normalised
        vehicles_front_left_lane = vehicles * np.hstack((rel_offset_front_left_lane, rel_vel_front_left_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_left_lane))

        # Right lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_right_lane = np.vstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneR"]["relB"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_right_lane = veh.s["laneR"]["relB"]["vel"][0] / max_vel  # Normalised
        vehicles_back_right_lane = vehicles * np.hstack((rel_offset_back_right_lane, rel_vel_back_right_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_back_right_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_right_lane = np.vstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                 veh.s["laneR"]["relF"]["gap"][0][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_right_lane = veh.s["laneR"]["relF"]["vel"][0] / max_vel  # Normalised
        vehicles_front_right_lane = vehicles * np.hstack((rel_offset_front_right_lane, rel_vel_front_right_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_right_lane))

        state = np.concatenate((np.squeeze(static_state), np.matrix.flatten(dynamic_state)))

        return state
    else:  # Remove velocities of other vehicles from state vector!

        D_MAX = 150

        # Static part of the state vector:
        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"] / (lane_width * 3)  # Normalised
        max_vel = veh.s["maxVel"]  # Excluded
        curr_vel = veh.s["vel"] / max_vel  # Normalised longitudinal component and lateral component excluded

        offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
        offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
        offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised

        static_state = np.hstack(
            (gap_to_road_edge, curr_vel, offset_current_lane_center, offset_left_lane_center, offset_right_lane_center))
        static_state = tf.expand_dims(tf.convert_to_tensor(static_state, dtype=tf.float32, name="static_state_input"),
                                      0)

        # Dynamic part of the state vector:
        # Current lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relB"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_back_center_lane = np.vstack((veh.s["laneC"]["relB"]["gap"][:, 0] / D_MAX,
                                                 veh.s["laneC"]["relB"]["gap"][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_back_center_lane = vehicles * rel_offset_back_center_lane
        dynamic_state = vehicles_back_center_lane

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relF"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_front_center_lane = np.vstack((veh.s["laneC"]["relF"]["gap"][:, 0] / D_MAX,
                                                  veh.s["laneC"]["relF"]["gap"][:,
                                                  1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_front_center_lane = vehicles * rel_offset_front_center_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_center_lane))

        # Left lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_left_lane = np.vstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / D_MAX,
                                               veh.s["laneL"]["relB"]["gap"][0][:,
                                               1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_back_left_lane = vehicles * rel_offset_back_left_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_back_left_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_left_lane = np.vstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneL"]["relF"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_front_left_lane = vehicles * rel_offset_front_left_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_left_lane))

        # Right lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_right_lane = np.vstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneR"]["relB"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default

        vehicles_back_right_lane = vehicles * rel_offset_back_right_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_back_right_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_right_lane = np.vstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                 veh.s["laneR"]["relF"]["gap"][0][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_front_right_lane = vehicles * rel_offset_front_right_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_right_lane))

        state = np.concatenate((np.squeeze(static_state), np.matrix.flatten(dynamic_state)))

        return state

def decompose_state(veh, remove_velocity, use_deepset=False):
    """
    Decomposes the state into static and dynamic components.
        Static components -> Ego vehicle speed, distance to road edge, etc.
        Dynamic components -> List of all information related to nearby vehicles.
    This is then passed to a permutation invariant neural network.
    Also normalises the state variables and excludes repeated or static variables.
    """
    if remove_velocity == False:
        D_MAX = 150

        # Static part of the state vector:
        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"] / (lane_width * 3)  # Normalised
        max_vel = veh.s["maxVel"]  # Excluded
        curr_vel = veh.s["vel"] / max_vel  # Normalised longitudinal component and lateral component excluded

        offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
        offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
        offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised

        static_state = np.hstack(
            (gap_to_road_edge, curr_vel, offset_current_lane_center, offset_left_lane_center, offset_right_lane_center))
        static_state = tf.expand_dims(tf.convert_to_tensor(static_state, dtype=tf.float32, name="static_state_input"), 0)

        # Dynamic part of the state vector:
        # Current lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relB"]["gap"][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_center_lane = np.vstack((veh.s["laneC"]["relB"]["gap"][:, 0] / D_MAX,
                                                 veh.s["laneC"]["relB"]["gap"][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_center_lane = veh.s["laneC"]["relB"]["vel"] / max_vel  # Normalised
        vehicles_back_center_lane = vehicles * np.hstack((rel_offset_back_center_lane, rel_vel_back_center_lane))
        dynamic_state = vehicles_back_center_lane

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relF"]["gap"][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_center_lane = np.vstack((veh.s["laneC"]["relF"]["gap"][:, 0] / D_MAX,
                                                  veh.s["laneC"]["relF"]["gap"][:,
                                                  1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_center_lane = veh.s["laneC"]["relF"]["vel"] / max_vel  # Normalised
        vehicles_front_center_lane = vehicles * np.hstack((rel_offset_front_center_lane, rel_vel_front_center_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_center_lane))

        # Left lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneL"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_left_lane = np.vstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / D_MAX,
                                               veh.s["laneL"]["relB"]["gap"][0][:,
                                               1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_left_lane = veh.s["laneL"]["relB"]["vel"][0] / max_vel  # Normalised
        vehicles_back_left_lane = vehicles * np.hstack((rel_offset_back_left_lane, rel_vel_back_left_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_back_left_lane))

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneL"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_left_lane = np.vstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneL"]["relF"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_left_lane = veh.s["laneL"]["relF"]["vel"][0] / max_vel  # Normalised
        vehicles_front_left_lane = vehicles * np.hstack((rel_offset_front_left_lane, rel_vel_front_left_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_left_lane))

        # Right lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneR"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_right_lane = np.vstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneR"]["relB"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_back_right_lane = veh.s["laneR"]["relB"]["vel"][0] / max_vel  # Normalised
        vehicles_back_right_lane = vehicles * np.hstack((rel_offset_back_right_lane, rel_vel_back_right_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_back_right_lane))

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneR"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_right_lane = np.vstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                 veh.s["laneR"]["relF"]["gap"][0][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        rel_vel_front_right_lane = veh.s["laneR"]["relF"]["vel"][0] / max_vel  # Normalised
        vehicles_front_right_lane = vehicles * np.hstack((rel_offset_front_right_lane, rel_vel_front_right_lane))
        dynamic_state = np.vstack((dynamic_state, vehicles_front_right_lane))

        dynamic_state = tf.expand_dims(tf.convert_to_tensor(dynamic_state, dtype=tf.float32, name="dynamic_state_input"), 0)

        return [dynamic_state, static_state]

    else:  # Remove velocities of other vehicles from state vector!
        D_MAX = 150

        # Static part of the state vector:
        lane_width = veh.s["laneC"]["width"]  # Excluded
        gap_to_road_edge = veh.s["gapB"] / (lane_width * 3)  # Normalised
        max_vel = veh.s["maxVel"]  # Excluded
        curr_vel = veh.s["vel"] / max_vel  # Normalised longitudinal component and lateral component excluded

        offset_current_lane_center = veh.s["laneC"]["off"] / lane_width  # Normalised
        offset_left_lane_center = veh.s["laneL"]["off"] / lane_width  # Normalised
        offset_right_lane_center = veh.s["laneR"]["off"] / lane_width  # Normalised

        static_state = np.hstack(
            (gap_to_road_edge, curr_vel, offset_current_lane_center, offset_left_lane_center, offset_right_lane_center))
        static_state = tf.expand_dims(tf.convert_to_tensor(static_state, dtype=tf.float32, name="static_state_input"),
                                      0)

        # Dynamic part of the state vector:
        # Current lane:
        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relB"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_back_center_lane = np.vstack((veh.s["laneC"]["relB"]["gap"][:, 0] / D_MAX,
                                                 veh.s["laneC"]["relB"]["gap"][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_back_center_lane = vehicles * rel_offset_back_center_lane
        dynamic_state = vehicles_back_center_lane

        vehicles = tf.expand_dims(tf.convert_to_tensor(veh.s["laneC"]["relF"]["gap"][:, 0] < D_MAX, dtype=tf.float32),
                                  1)
        rel_offset_front_center_lane = np.vstack((veh.s["laneC"]["relF"]["gap"][:, 0] / D_MAX,
                                                  veh.s["laneC"]["relF"]["gap"][:,
                                                  1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_front_center_lane = vehicles * rel_offset_front_center_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_center_lane))

        # Left lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_left_lane = np.vstack((veh.s["laneL"]["relB"]["gap"][0][:, 0] / D_MAX,
                                               veh.s["laneL"]["relB"]["gap"][0][:,
                                               1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_back_left_lane = vehicles * rel_offset_back_left_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_back_left_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneL"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_left_lane = np.vstack((veh.s["laneL"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneL"]["relF"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_front_left_lane = vehicles * rel_offset_front_left_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_left_lane))

        # Right lane:
        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relB"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_back_right_lane = np.vstack((veh.s["laneR"]["relB"]["gap"][0][:, 0] / D_MAX,
                                                veh.s["laneR"]["relB"]["gap"][0][:,
                                                1] / lane_width)).transpose()  # Normalised to Dmax default
        vehicles_back_right_lane = vehicles * rel_offset_back_right_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_back_right_lane))

        vehicles = tf.expand_dims(
            tf.convert_to_tensor(veh.s["laneR"]["relF"]["gap"][0][:, 0] < D_MAX, dtype=tf.float32), 1)
        rel_offset_front_right_lane = np.vstack((veh.s["laneR"]["relF"]["gap"][0][:, 0] / D_MAX,
                                                 veh.s["laneR"]["relF"]["gap"][0][:,
                                                 1] / lane_width)).transpose()  # Normalised to Dmax default

        vehicles_front_right_lane = vehicles * rel_offset_front_right_lane
        dynamic_state = np.vstack((dynamic_state, vehicles_front_right_lane))

        dynamic_state = tf.expand_dims(
            tf.convert_to_tensor(dynamic_state, dtype=tf.float32, name="dynamic_state_input"), 0)

        return [dynamic_state, static_state]


class RewardFunction(object):

    def _get_velocity_reward(self, current_velocity):
        max_vel = 120 / 3.6
        reward_vel = np.exp(-(max_vel - current_velocity) ** 2 / 300)
        return reward_vel

    def _get_lane_centre_reward(self, lane_offset):
        reward_offset = np.exp(-(lane_offset) ** 2 / 1.18)
        return reward_offset

    def _get_follow_dist_reward(self, following_distance):
        distance_limit = 0
        reward_following_distance = -np.exp(-(distance_limit - following_distance) ** 2 / 250)
        return reward_following_distance

    def _get_right_lane_reward(self, total_road_width, dist_to_edge):
        reward_right_lane = ((1 - 0) / (0 - total_road_width)) * dist_to_edge + 1
        # reward_right_lane = (1 / total_road_width) * dist_to_edge  # Stay left
        return reward_right_lane

    def get_reward(self, agent, veh=None):
        """
        Calculate reward for actions.
        Reward = Speed + LaneCentre + FollowingDistance
        """
        # Reward function weightings:
        w_vel = agent.reward_weights[0]  # Speed weight
        w_off = agent.reward_weights[1]  # Lane center
        w_dist = agent.reward_weights[2]  # Following distance
        w_stay_right = agent.reward_weights[3]  # Staying in right lane
        weights_sum = np.sum(np.array([w_vel, w_off, w_dist, w_stay_right]))
        w_vel = w_vel/weights_sum
        w_off = w_off/weights_sum
        w_dist = w_dist/weights_sum
        w_stay_right = w_stay_right/weights_sum

        # Reward function declaration:
        reward = np.array([0], dtype=np.float32)
        if veh is not None:
            # Velocity reward:
            v = np.squeeze(veh.s["vel"])[0]
            r_vel = self._get_velocity_reward(current_velocity=v)

            # TODO remove lane centre reward when acting with discrete lane changes
            # Lane center reward:
            lane_offset = np.squeeze(veh.s["laneC"]["off"])
            r_off = self._get_lane_centre_reward(lane_offset=lane_offset)

            # Following distance:
            d_gap = veh.s["laneC"]["relF"]["gap"][0, 0]
            r_follow = self._get_follow_dist_reward(following_distance=d_gap)

            # Stay right:
            road_width = veh.s["laneC"]["width"] * 3
            road_edge_dist = veh.s["gapB"][0]
            r_right = self._get_right_lane_reward(total_road_width=road_width, dist_to_edge=road_edge_dist)

            reward = (w_vel * r_vel + w_off * r_off + w_dist * r_follow + w_stay_right * r_right)
            reward_max = w_vel * 1.0 + w_off * 1.0 + w_dist * 0.0 + w_stay_right * 1.0
            reward_min = w_vel * 0.0 + w_off * 0.0 + w_dist * -1.0 + w_stay_right * 0.0
            reward = (reward - reward_min) / (reward_max - reward_min)
        else:
            reward = 0
        return reward

    def plot_reward_functions(self):
        plt.ion()

        fig, axs = plt.subplots(2, 2)
        vel = np.linspace(0, 130 / 3.6, 100)
        rew = self._get_velocity_reward(vel)
        axs[0, 0].plot(vel, rew)
        axs[0, 0].set_title('Velocity')
        axs[0, 0].grid(True)
        axs[0, 0].set(xlabel='Velocity', ylabel='Reward')

        y = np.linspace(-3.6, 3.6, 100)
        rew = self._get_lane_centre_reward(y)
        axs[0, 1].plot(y, rew, 'tab:orange')
        axs[0, 1].set_title('Lane Centre')
        axs[0, 1].grid(True)
        axs[0, 1].set(xlabel='Lane position', ylabel='Reward')

        x = np.linspace(0, 50, 100)
        rew = self._get_follow_dist_reward(x)
        axs[1, 0].plot(x, rew, 'tab:green')
        axs[1, 0].set_title('Following Distance')
        axs[1, 0].grid(True)
        axs[1, 0].set(xlabel='Following distance', ylabel='Reward')

        road_width = 12
        y = np.linspace(0, road_width, 100)
        rew = self._get_right_lane_reward(road_width, y)
        axs[1, 1].plot(y, rew, 'tab:red')
        axs[1, 1].set_title('Lane Position')
        axs[1, 1].grid(True)
        axs[1, 1].set(xlabel='Distance from right road edge', ylabel='Reward')

        fig.tight_layout()
        plt.show()


class DiscreteSingleActionPolicy(CustomPolicy):
    """
    DDQN!!!!
    """
    LONG_ACTION = ActionType.REL_VEL
    LAT_ACTION = ActionType.LANE

    def __init__(self, agent):
        super(DiscreteSingleActionPolicy, self).__init__()
        self.agent = agent  # agent = f(NN_model)
        self.STEP_TIME = self.agent.training_param["policy_rate"]
        self.rewards = RewardFunction()
        if self.agent.training_param['use_temporal_CNN'] or self.agent.training_param["use_LSTM"]:
            self.stack_frames = True
            if self.agent.training_param['use_temporal_CNN']:
                self.stack_cnn_frames = True
                self.stack_LSTM_frames = False
            elif self.agent.training_param["use_LSTM"]:
                self.stack_cnn_frames = False
                self.stack_LSTM_frames = True
            self.frame_stack_buffer = deque(maxlen=4)
        else:
            self.stack_frames = False
        self.remove_velocity = self.agent.training_param["remove_state_vel"]


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
        self.agent.is_action_taken = False

    def custom_action(self, veh):
        # s0, a0 = previous vehicle state action pair
        # s1, a1 = current vehicle state action pair
        veh.counter -= 1

        if veh.counter <= 0:
            veh.counter = self.STEP_TIME
            # Set current vehicle state and action pair
            veh.s1 = veh.s_raw
            if self.agent.training_param["use_deepset"] or self.agent.training_param["use_CNN"] or self.agent.training_param['use_temporal_CNN']:
                veh.s1_mod = decompose_state(veh,
                                             remove_velocity=self.remove_velocity,
                                             use_deepset=self.agent.training_param["use_deepset"])
            else:
                veh.s1_mod = convert_state(veh, remove_velocity=self.remove_velocity)


            if self.agent.training_param["noise_param"]["use_noise"]:
                # TODO !!!!!!!!!!! ONLY ADD STATE NOISE ON NON-ZERO MEASUREMENTS !!!!!!!!!!!!!!
                veh.s1_mod = add_state_noise(veh.s1_mod,
                                             is_normal=self.agent.training_param["noise_param"]["normal"],
                                             is_uniform=self.agent.training_param["noise_param"]["uniform"],
                                             magnitude=self.agent.training_param["noise_param"]["magnitude"],
                                             mu=self.agent.training_param["noise_param"]["mu"],
                                             sigma=self.agent.training_param["noise_param"]["sigma"])

            if self.stack_frames:
                if self.stack_LSTM_frames:
                    current_buffer_size = len(self.frame_stack_buffer)
                    if current_buffer_size < 3:
                        for item in range(3-current_buffer_size):
                            self.frame_stack_buffer.append(veh.s1_mod)
                    self.frame_stack_buffer.append(veh.s1_mod)
                    stacked_state = tf.expand_dims(tf.convert_to_tensor(list(self.frame_stack_buffer)), 0)
                    Q = self.get_action(stacked_state)
                    veh.s1_mod = stacked_state
                if self.stack_cnn_frames:
                    current_buffer_size = len(self.frame_stack_buffer)
                    if current_buffer_size < 3:
                        for item in range(3-current_buffer_size):
                            self.frame_stack_buffer.append(veh.s1_mod[0])
                    self.frame_stack_buffer.append(veh.s1_mod[0])
                    dynamic_stacked_state = tf.convert_to_tensor(list(self.frame_stack_buffer))
                    # Transpose into shape suitable for the CNN
                    dynamic_stacked_state = tf.transpose(dynamic_stacked_state, perm=[1,0,2,3])
                    full_stacked_state = [dynamic_stacked_state,
                                             veh.s1_mod[1]]
                    Q = self.get_action(full_stacked_state)
                    veh.s1_mod = full_stacked_state
            else:  # No frame stacking
                if self.agent.training_param["use_deepset"] or self.agent.training_param["use_CNN"]:
                    Q = self.get_action(veh.s1_mod)
                else:
                    veh.s1_mod = tf.expand_dims(veh.s1_mod, axis=0)
                    Q = self.get_action(veh.s1_mod)



            veh.a1_mod = Q
            action_choice = self.agent.get_action_choice(Q)
            veh.a1_choice = action_choice
            veh.a1 = self.convert_action_discrete(veh, action_choice)

            # Save experience
            if veh.a0_mod is not None:  # Check if the agent has taken an action that led to this reward...
                # Calculate reward at current state (if action was taken previously)
                veh.flag = self.agent.stop_flags
                veh.reward = (self.rewards.get_reward(agent=self.agent, veh=veh) + np.sum(veh.rew_buffer))/self.STEP_TIME
                self.agent.latest_reward = veh.reward
                if self.agent is not None and self.agent.training is True:
                    # Save action taken previously on previous state value
                    experience = (veh.s0_mod,
                                  veh.a0_choice,
                                  veh.reward,
                                  veh.s1_mod)
                    self.agent.latest_experience = experience
                    self.agent.is_action_taken = True

            # Set past vehicle state and action pair
            veh.s0 = veh.s1
            veh.s0_mod = veh.s1_mod
            veh.a0 = veh.a1
            veh.a0_mod = veh.a1_mod
            veh.a0_choice = veh.a1_choice
            veh.c0 = veh.c1
            veh.rew_buffer = []
            output = np.array([veh.a1[0], veh.a1[1]],
                              dtype=np.float64)  # The hwsim library uses double precision floats
            veh.prev_action = action_choice
            veh.LONG_ACTION = output[0]
            veh.LAT_ACTION = output[1]
            return output

        else:
            self.agent.is_action_taken = False
            discrete_actions = self.convert_action_discrete(veh, veh.prev_action)
            output = np.array(discrete_actions, dtype=np.float64)
            veh.rew_buffer.append(self.rewards.get_reward(agent=self.agent, veh=veh))
            return output

    def get_action(self, state):
        """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
        # Receive action and critic values from NN:
        # with tf.device('/GPU:1'):
        Q = self.agent.Q_actual_net(state)
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
        # [lower bound, upper bound]
        # right lane = [0,1]
        # left lane = [-1,0]
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
            steer_controller = np.minimum(off_bounds[1], 1)
        elif action_choices == 4:  # Constant speed, turn right
            vel_controller = np.minimum(vel_bounds[1], 0)
            steer_controller = np.maximum(off_bounds[0], -1)
        else:
            print("Error with setting offset action!")

        sim_action[0] = vel_controller
        sim_action[1] = steer_controller
        return sim_action


class pyStepPolicy(CustomPolicy):
    MIN_REL_OFF = 0.0
    MAX_REL_OFF = 1.0

    def init_vehicle(self, veh):
        veh.k = 0
        veh.curActions = None
        self.period = random.randint(25, 100)
        self.vr = [random.uniform(0.1, 0.3), random.uniform(0.5, 0.7)]


    def custom_action(self, veh):
        if veh.k<=0:
            veh.k = self.period
            veh.curActions = (random.uniform(self.vr[0], self.vr[1]), random.uniform(self.MIN_REL_OFF, self.MAX_REL_OFF))
        veh.k -= 1
        a_bounds = veh.a_bounds
        vel = veh.curActions[0]*(a_bounds["long"][1]-a_bounds["long"][0]) + a_bounds["long"][0]
        off = veh.curActions[1]*(a_bounds["lat"][1]-a_bounds["lat"][0]) + a_bounds["lat"][0]
        return np.array([vel,off])

########################################################################


# class DiscreteDoubleActionPolicy(CustomPolicy):
#     """
#     Actor-critic on-policy RL controller for highway decision making.
#     """
#     LONG_ACTION = ActionType.REL_VEL
#     LAT_ACTION = ActionType.LANE
#
#     def __init__(self, agent):
#         super(DiscreteDoubleActionPolicy, self).__init__()
#         self.agent = agent  # agent = f(NN_model)
#         self.STEP_TIME = self.agent.training_param["STEP_TIME"]
#         self.rewards = RewardFunction()
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
#             veh.s1_mod = convert_state(veh)
#             action_probs, veh.c1 = self.get_action_and_critic(veh.s1_mod)
#             self.agent.states.append(veh.s1_mod)
#             self.agent.actions.append(action_probs)
#             veh.a1_mod = action_probs
#             action_choice = self.agent.get_action_choice(action_probs)
#             veh.a1_choice = action_choice
#             veh.a1 = self.convert_action_discrete(veh, action_choice)
#
#             # Save experience
#             if veh.a0_mod is not None:  # Check if the agent has taken an action that led to this reward...
#                 # Calculate reward at current state (if action was taken previously)
#                 veh.reward = self.rewards.get_reward(agent=self.agent, veh=veh) + np.sum(veh.rew_buffer)
#                 self.agent.rewards.append(veh.reward)
#                 if self.agent is not None and self.agent.training is True:
#                     # Save action taken previously on previous state value
#                     # action = velocity action, lane_change action
#                     action = veh.a0_mod[0][veh.a0_choice]
#
#                     # Save to buffer from the Buffer class in HelperClasses.py module
#                     # add_experience expects (timestep, state, vel_model_action, off_model_action,
#                     #                         vel_action_sim, offset_action_sim, vel_choice, off_choice, reward, critic)
#                     self.agent.buffer.set_experience(timestep=self.agent.timestep,
#                                                      state=np.squeeze(veh.s0_mod),
#                                                      action_model=np.array(action),
#                                                      action_sim=veh.a0,
#                                                      action_choice=veh.a0_choice,
#                                                      reward=veh.reward,
#                                                      critic=np.squeeze(veh.c0))
#
#             # Set past vehicle state and action pair
#             veh.s0 = veh.s1
#             veh.s0_mod = veh.s1_mod
#             veh.a0 = veh.a1
#             veh.a0_mod = veh.a1_mod
#             veh.a0_choice = veh.a1_choice
#             veh.c0 = veh.c1
#             veh.rew_buffer = []
#             output = np.array([veh.a1[0], veh.a1[1]],
#                               dtype=np.float64)  # The hwsim library uses double precision floats
#             veh.prev_action = action_choice
#             return output
#
#         else:
#             discrete_actions = self.convert_action_discrete(veh, veh.prev_action)
#             output = np.array(discrete_actions, dtype=np.float64)
#             veh.rew_buffer.append(self.rewards.get_reward(agent=self.agent, veh=veh))
#             return output
#
#     def get_action_and_critic(self, state):
#         """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
#         # Receive action and critic values from NN:
#         action_probs, critic = self.agent.actor_critic_net(state)
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
#
# class FixedLanePolicy(CustomPolicy, enc_name="fixed_lane"):
#     """ Simple policy where each vehicle will stay in its initial lane with a certain target
#     velocity (relative to the maximum allowed speed). The actual velocity is always upper
#     bounded by the safety bounds (taking vehicles in front into account)."""
#     LONG_ACTION = ActionType.ABS_VEL
#     LAT_ACTION = ActionType.LANE  # Alternatively: ActionType.LANE
#
#     def __init__(self, speed):
#         super().__init__()
#         self.STEP_TIME = 100  # Change reference velocity every 100 iterations (10s)
#         self.speed = speed
#
#     def init_vehicle(self, veh):
#         """ Policy objects are shared over many different vehicles so to associate
#         attributes to specific vehicles, we can use this method (which is called
#         during Vehicle instantiation) """
#         veh.abs_vel = self.speed
#         veh.counter = 0
#
#     def _set_rel_vel(self, veh):
#         veh.rel_vel = 0.95 - random.random() * 0.3
#
#     def custom_action(self, veh):
#         """ This method is called at every iteration and the returned numpy arrary
#         will be used as the new reference actions (passed to the lower level controllers
#         who will set up proper model inputs to track the new reference) """
#         # Start with updating the counter and setting a new reference if necessary
#         veh.counter -= 1
#         if veh.counter <= 0:
#             veh.counter = self.STEP_TIME
#             self._set_rel_vel(veh)
#         # Then calculate proper actions from the current reference
#         s = veh.s  # Current augmented state
#         bounds = veh.a_bounds  # Current safety bounds on the actions (calculated from the current augmented state). Vehicle operation remains 'safe' as long as we respect these bounds.
#         v_max = veh.rel_vel * (s["maxVel"])
#         v = min(v_max, bounds["long"][1])
#         v = max(0, v)
#         # Final actions are: the target velocity and negating the offset towards the lane center
#         return np.array([v, 0])
#         # Alternatively (with LANE actionType):
#         # return np.array([v,0]) # Lane reference is 0 => remain in (center of) current lane
