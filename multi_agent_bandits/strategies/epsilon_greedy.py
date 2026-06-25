import random

import numpy as np

from multi_agent_bandits.core.agent import Agent


class EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms, epsilon=0.1, name=None, seed=None):
        super().__init__(n_arms, name=name)
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None

        # track estimates and counts
        self.counts = [0] * n_arms
        # Initialize with small random values to break symmetry
        if self.rng is not None:
            self.values = self.rng.uniform(-0.01, 0.01, n_arms)
        else:
            self.values = np.random.uniform(-0.01, 0.01, n_arms)
        self.last_arm = None

    def choose_arm(self):
        if self.rng is not None:
            if self.rng.random() < self.epsilon:
                self.last_arm = int(self.rng.integers(self.n_arms))
                return self.last_arm
        elif random.random() < self.epsilon:
            self.last_arm = random.randrange(self.n_arms)
            return self.last_arm

        self.last_arm = int(np.argmax(self.values))
        return self.last_arm

    def update(self, reward):
        # update the strategy with new info
        arm = self.last_arm

        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])
