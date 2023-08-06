import random
from collections import deque

import numpy as np


class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

        self.is_over = False

    def add_timestep(self, timestep):
        self.observations.append(timestep.observation)

        if not timestep.first():
            self.actions.append(timestep.action)
            self.rewards.append(timestep.reward)

        if timestep.last():
            self.is_over = True

    def __len__(self):
        return len(self.observations) - 1


class ReplayMemory:
    def __init__(self, max_size):
        self.curr_episode = Episode()
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_observations = deque(maxlen=max_size)

    def add_timestep(self, timestep):
        self.curr_episode.add_timestep(timestep)
        if self.curr_episode.is_over:
            self.observations.extend(self.curr_episode.observations[:-1])
            self.actions.extend(self.curr_episode.actions)
            self.rewards.extend(self.curr_episode.rewards)
            self.next_observations.extend(self.curr_episode.observations[1:])
            self.curr_episode = Episode()

    def sample_steps(self, size):
        indices = list(range(len(self.observations)))
        random.shuffle(indices)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        for i in range(min(size, len(indices))):
            index = indices[i]
            observations.append(self.observations[index])
            actions.append(self.actions[index])
            rewards.append(self.rewards[index])
            next_observations.append(self.next_observations[index])

        observations = np.stack(observations)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_observations = np.stack(next_observations)
        return observations, actions, rewards, next_observations


