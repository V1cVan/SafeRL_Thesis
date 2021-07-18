import numpy as np
from collections import deque
import tensorflow as tf
import random
from HelperClasses import Timer


class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, buffer_size, batch_size, use_deepset_or_cnn=False, stack_frames=False):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.use_deepset_or_cnn = use_deepset_or_cnn
        self.stack_frames = stack_frames

    def add_experience(self, experience):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        if self.use_deepset_or_cnn:
            states, actions, rewards, next_states, done = experience
            dynamic_states = np.squeeze(states[0])
            static_states = np.squeeze(states[1])
            dynamic_next_states = np.squeeze(next_states[0])
            static_next_states = np.squeeze(next_states[1])
            experience = (dynamic_states, static_states,
                          actions, rewards,
                          dynamic_next_states, static_next_states,
                          done)
            self.buffer.append(experience)
        else:
            states, actions, rewards, next_states, done = experience
            experience = (np.squeeze(states), actions, rewards, np.squeeze(next_states), done)
            self.buffer.append(experience)

    def get_training_samples(self):
        """ Get mini-batch for training. """
        number_stacked_frames = 4
        if self.stack_frames:
            """If frames are stacked additional samples are extracted from the buffer to capture temporal variations"""
            # Sample random indices
            indices = range(len(self.buffer))
            mini_batch_indices = random.sample(indices[number_stacked_frames-1:], self.batch_size)
            stacked_index_matrix = tf.expand_dims(mini_batch_indices, axis=1) + tf.convert_to_tensor(np.array([[-3, -2, -1, 0]]))
            # Flatten index_matrix and extract with stacked frames:
            stacked_mini_batch = [self.buffer[index] for index in tf.reshape(stacked_index_matrix, [-1])]

            # Extract normal mini batch
            mini_batch = [self.buffer[index] for index in mini_batch_indices]

            # Check if mini_batch values include terminal states and remove them.
            # TODO Resample instead of discarding if stacked frame contains terminal state
            if self.use_deepset_or_cnn:
                done = [each[6] for each in stacked_mini_batch]
            else:
                done = [each[4] for each in stacked_mini_batch]
            done = tf.reshape(done, stacked_index_matrix.shape)
            mask = tf.cast(tf.reduce_all(tf.math.logical_not(done), axis=1, keepdims=True), dtype=tf.float32)
            mask = mask*np.array([[1, 1, 1, 1]])  # Convert mask to matrix instead of vector
            stacked_index_matrix = tf.boolean_mask(stacked_index_matrix, mask)
            stacked_index_matrix = tf.reshape(stacked_index_matrix, tf.shape(mask))
            stacked_mini_batch = [self.buffer[index] for index in tf.reshape(stacked_index_matrix, [-1])]
            altered_batch_size = stacked_index_matrix.shape[0]
        else:
            # TODO ! Different frames for the LSTM network
            mini_batch = random.sample(self.buffer, self.batch_size)


        # TODO Remove additional for loops to speed up training!!
        if self.use_deepset_or_cnn:
            dynamic_states = []
            static_states = []
            actions = []
            rewards = []
            dynamic_next_states = []
            static_next_states = []
            done = []
            if self.stack_frames:
                for each in stacked_mini_batch:
                    dynamic_states.append(each[0])
                    static_states.append(each[1])
                    dynamic_next_states.append(each[4])
                    static_next_states.append(each[5])
                for each in mini_batch:
                    actions.append(each[2])
                    rewards.append(each[3])
                    done.append(each[6])
            else:
                for each in mini_batch:
                    dynamic_states.append(each[0])
                    static_states.append(each[1])
                    actions.append(each[2])
                    rewards.append(each[3])
                    dynamic_next_states.append(each[4])
                    static_next_states.append(each[5])
                    done.append(each[6])

            dynamic_states = tf.squeeze(tf.convert_to_tensor(dynamic_states, dtype=tf.float32))
            static_states = tf.squeeze(tf.convert_to_tensor(static_states, dtype=tf.float32))
            dynamic_next_states = tf.squeeze(tf.convert_to_tensor(dynamic_next_states, dtype=tf.float32))
            static_next_states = tf.squeeze(tf.convert_to_tensor(static_next_states, dtype=tf.float32))
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=tf.float32))
            rewards = tf.squeeze(tf.convert_to_tensor(rewards, dtype=tf.float32))
            done = tf.cast(done, dtype=tf.float32)

            # Convert stacked frames into shape suitable for CNN networks:
            if self.stack_frames:

                dynamic_states = tf.reshape(dynamic_states, (altered_batch_size,
                                                             dynamic_states.shape[1],
                                                             dynamic_states.shape[2],
                                                             number_stacked_frames))
                static_states = tf.reshape(static_states, (altered_batch_size,
                                                           static_states.shape[1],
                                                           number_stacked_frames))
                dynamic_next_states = tf.reshape(dynamic_next_states, (altered_batch_size,
                                                                       dynamic_next_states.shape[1],
                                                                       dynamic_next_states.shape[2],
                                                                        number_stacked_frames))
                static_next_states = tf.reshape(static_next_states, (altered_batch_size,
                                                                     static_next_states.shape[1],
                                                                     number_stacked_frames))

            states = (dynamic_states, static_states)
            next_states = (dynamic_next_states, static_next_states)
            # TODO check if the implemented method actually improves speed
            # dynamic_states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
            # static_states = tf.squeeze(tf.convert_to_tensor([each[1] for each in mini_batch], dtype=np.float32))
            # states = (dynamic_states, static_states)
            #
            # actions = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch])))
            # rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
            #
            # dynamic_next_states = tf.squeeze(tf.convert_to_tensor([each[4] for each in mini_batch], dtype=np.float32))
            # static_next_states = tf.squeeze(tf.convert_to_tensor([each[5] for each in mini_batch], dtype=np.float32))
            # next_states = (dynamic_next_states, static_next_states)
            #
            # done = tf.cast([each[6] for each in mini_batch], dtype=tf.float32)

        else: # If not using Deepset or CNN network architectures
            states = []
            actions = []
            rewards = []
            next_states = []
            done = []

            if self.stack_frames:
                for each in stacked_mini_batch:
                    states.append(each[0])
                    next_states.append(each[3])
                for each in mini_batch:
                    actions.append(each[1])
                    rewards.append(each[2])
                    done.append(each[4])
            else: # If not using architectures that capture temporal information using stacked frames
                for each in mini_batch:
                    states.append(each[0])
                    actions.append(each[1])
                    rewards.append(each[2])
                    next_states.append(each[3])
                    done.append(each[4])

            states = tf.squeeze(tf.convert_to_tensor(states, dtype=tf.float32))
            next_states = tf.squeeze(tf.convert_to_tensor(next_states, dtype=tf.float32))
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=tf.float32))
            rewards = tf.squeeze(tf.convert_to_tensor(rewards, dtype=tf.float32))
            done = tf.cast(done, dtype=tf.float32)

            # Reshape state matrices into form suitable for the LSTM networks
            if self.stack_frames:
                states = tf.reshape(states, (altered_batch_size,
                                             states.shape[1],
                                             number_stacked_frames))
                next_states = tf.reshape(next_states, (altered_batch_size,
                                                       next_states.shape[1],
                                                       number_stacked_frames))
            # TODO check if it works
            # states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
            # actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch])))
            # rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch], dtype=np.float32)))
            # next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
            # done = tf.cast([each[4] for each in mini_batch], dtype=tf.float32)

        return states, actions, rewards, next_states, done

    def alter_buffer_stop_flag(self, flag):
        state, action, reward, next_state, done_flag = self.buffer[-1]
        done_flag = flag
        self.buffer[-1] = state, action, reward, next_state, done_flag

    def is_buffer_min_size(self):
        return self.get_size() > self.batch_size+64

    def get_size(self):
        return len(self.buffer)


class PerTrainingBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, buffer_size, batch_size, alpha, beta, beta_increment, use_deepset_or_cnn = False, stack_frames=False):
        self.tree = SumTree(buffer_size)
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.e = 0.01
        self.use_deepset_or_cnn = use_deepset_or_cnn
        self.stack_frames = stack_frames

    def alter_buffer_stop_flag(self, flag):
        done_flag = flag
        # state, action, reward, next_state, done_flag = self.buffer[-1]
        # done_flag = flag
        # self.buffer[-1] = state, action, reward, next_state, done_flag

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def add_experience(self, error, experience):
        p = self._get_priority(error)
        self.tree.add(p, experience)

    def get_training_samples(self):
        mini_batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            mini_batch.append(data)
            idxs.append(idx)

        # TODO fix the division by self.tree.total() and rather just divide by the value of the root node
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = tf.squeeze(tf.convert_to_tensor(is_weight, dtype=np.float32))

        # Get minibatch for training
        # TODO Remove additional for loops to speed up training like above!
        if self.use_deepset_or_cnn:
            dynamic_states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
            static_states = tf.squeeze(tf.convert_to_tensor([each[1] for each in mini_batch], dtype=np.float32))
            states = (dynamic_states, static_states)

            actions = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch])))
            rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))

            dynamic_next_states = tf.squeeze(tf.convert_to_tensor([each[4] for each in mini_batch], dtype=np.float32))
            static_next_states = tf.squeeze(tf.convert_to_tensor([each[5] for each in mini_batch], dtype=np.float32))
            next_states = (dynamic_next_states, static_next_states)

            done = tf.cast([each[6] for each in mini_batch], dtype=tf.float32)
        else:
            states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
            actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch])))
            rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch], dtype=np.float32)))
            next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
            done = tf.cast([each[4] for each in mini_batch], dtype=tf.float32)

        return states, actions, rewards, next_states, done, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        for idx, p in zip(idx, p):
            self.tree.update(idx, p)

    def is_buffer_min_size(self):
        return self.get_size() > self.batch_size+64

    def get_size(self):
        return self.tree.n_entries


class SumTree:
    # Source = https://github.com/jcborges/dqn-per/blob/master/Memory.py
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


