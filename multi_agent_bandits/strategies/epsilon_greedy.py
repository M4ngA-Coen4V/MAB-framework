import random
from multi_agent_bandits.core.agent import Agent

class EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms, epsilon=0.1, name = None):
        super().__init__(n_arms, name=name)
        self.epsilon = epsilon

        #track estimates and counts
        self.counts = [0] * n_arms
        #Initialize with small random values to break symmetry
        self.values = [random.uniform(0, 0.01) for _ in range(n_arms)]
        self.last_arm = None

    def choose_arm(self):
        if random.random() < self.epsilon:
            self.last_arm = random.randrange(self.n_arms)
            return self.last_arm

        self.last_arm = max(range(self.n_arms), key=lambda a: self.values[a])
        return self.last_arm

    def update(self, reward):
        #update the strategy with new info
        arm = self.last_arm
        
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm]) 
