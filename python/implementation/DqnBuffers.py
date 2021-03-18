import numpy as np
from collections import deque
import tensorflow as tf
import random


class TrainingBuffer(object):
    """
    The training buffer is used to store experiences that are then sampled from uniformly to facilitate
    improved training. The training buffer reduces the correlation between experiences and avoids that
    the network 'forgets' good actions that it learnt previously.
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_experience(self, experience):
        """
        Add an experience (s_k, a_k, r_k, s_k+1) to the training buffer.
        """
        self.buffer.append(experience)

    def get_training_samples(self):
        """ Get minibatch for training. """
        mini_batch = random.sample(self.buffer, self.batch_size)
        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in mini_batch], dtype=np.float32))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in mini_batch])))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in mini_batch], dtype=np.float32)))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in mini_batch], dtype=np.float32)))
        done = tf.cast([each[4] for each in mini_batch], dtype=tf.float32)
        return states, actions, rewards, next_states, done

    def alter_buffer_stop_flag(self, flag):
        state, action, reward, next_state, done_flag = self.buffer[-1]
        done_flag = flag
        self.buffer[-1] = state, action, reward, next_state, done_flag

    def is_buffer_min_size(self):
        return self.get_size() >= self.batch_size

    def get_size(self):
        return len(self.buffer)


class PerTrainingBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, buffer_size, batch_size, alpha, beta, beta_increment):
        self.tree = SumTree(buffer_size)
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.e = 0.01

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
        batch = []
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
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = tf.squeeze(tf.convert_to_tensor(is_weight, dtype=np.float32))

        states = tf.squeeze(tf.convert_to_tensor([each[0] for each in batch], dtype=np.float32))
        actions = tf.squeeze(tf.convert_to_tensor(np.array([each[1] for each in batch], dtype=np.float32)))
        rewards = tf.squeeze(tf.convert_to_tensor(np.array([each[2] for each in batch], dtype=np.float32)))
        next_states = tf.squeeze(tf.convert_to_tensor(np.array([each[3] for each in batch], dtype=np.float32)))
        done = tf.cast([each[4] for each in batch], dtype=tf.float32)

        return states, actions, rewards, next_states, done, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def is_buffer_min_size(self):
        if self.tree.n_entries < self.batch_size:
            return False
        else:
            return True

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


