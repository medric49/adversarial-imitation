import random
from collections import deque

import numpy as np


class Episode:
    def __init__(self, discount):
        self.discount = discount
        self.observations = []
        self.actions = []
        self.rewards = []

        self.is_over = False

    def add_timestep(self, timestep):
        self.observations.append(timestep.observation)

        if not timestep.first():
            self.actions.append(timestep.action)
            self.rewards.append(timestep.reward)
            if timestep.reward != 0:
                print(timestep.reward)

        if timestep.last():
            self.is_over = True

    @property
    def rewards_to_go(self):
        rewards_to_go = []
        for i in range(len(self)):
            reward_to_go = sum([reward * (self.discount ** j) for j, reward in enumerate(self.rewards[i:])])
            rewards_to_go.append(reward_to_go)
        return rewards_to_go

    def __len__(self):
        return len(self.observations) - 1


class ReplayMemory:
    def __init__(self, max_size, discount):
        self.discount = discount
        self.curr_episode = Episode(self.discount)

        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.rewards_to_go = deque(maxlen=max_size)
        self.next_observations = deque(maxlen=max_size)

    def add_timestep(self, timestep):
        self.curr_episode.add_timestep(timestep)
        if self.curr_episode.is_over:
            self.observations.extend(self.curr_episode.observations[:-1])
            self.actions.extend(self.curr_episode.actions)
            self.rewards.extend(self.curr_episode.rewards)
            self.rewards_to_go.extend(self.curr_episode.rewards_to_go)
            self.next_observations.extend(self.curr_episode.observations[1:])
            self.curr_episode = Episode(self.discount)

    def sample_steps(self, size, reward_to_go):
        indices = list(range(len(self.observations)))
        random.shuffle(indices)
        indices = indices[:size]
        observations = np.stack(self.observations)[indices]
        actions = np.stack(self.actions)[indices]
        if reward_to_go:
            rewards = np.stack(self.rewards_to_go)[indices]
        else:
            rewards = np.stack(self.rewards)[indices]
        next_observations = np.stack(self.next_observations)[indices]
        return observations, actions, rewards, next_observations
