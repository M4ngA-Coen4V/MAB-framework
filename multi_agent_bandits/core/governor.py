import math
import random


class LearningGovernorAI:
    """
    Continuous-action governor with a simple policy-gradient skeleton.
    
    This agent uses a Gaussian policy (normal distribution) to output continuous
    tax/subsidy adjustments for each individual agent. It optimizes its weights 
    using a custom REINFORCE-style policy gradient algorithm to maximize collective
    social rewards while penalizing system deaths.
    """

    def __init__(self, n_agents=3, action_std=3.0, learning_rate=0.01, death_penalty=0.0, seed=None):
        """
        Initialize the Learning Governor AI.

        Args:
            n_agents (int): Total number of agents in the system.
            action_std (float): Standard deviation (exploration width) for the Gaussian policy.
            learning_rate (float): Step size for the policy gradient parameter updates.
            death_penalty (float): Negative penalty per agent death in the governor reward.
            seed (int, optional): Random seed for reproducibility.
        """
        self.n_agents = n_agents
        self.action_std = action_std
        self.learning_rate = learning_rate
        self.death_penalty = death_penalty
        
        # Policy parameters: The mean (mu) value for each agent's redistribution profile.
        # Initialized to 0.0, meaning the governor starts with a completely neutral policy.
        self.means = [0.0] * n_agents
        
        # Memory storage buffers for step-by-step reinforcement learning updates
        self.last_observation = None
        self.last_action = None
        self.last_log_prob = None
        self.history = []

        if seed is not None:
            random.seed(seed)

    def choose_action(self, observation):
        """
        Return a raw continuous adjustment vector for the agents.
        Each element represents an un-normalized tax or subsidy proposal.

        Args:
            observation (list): Current state of the environment (agent choices + wealth levels).

        Returns:
            list: Raw continuous vector of size n_agents sampled from the policy distributions.
        """
        self.last_observation = observation
        
        # Sample raw adjustments from normal distributions centered around self.means
        action = [random.gauss(mu, self.action_std) for mu in self.means]
        
        # Cache current actions and policy log probabilities for the gradient calculation later
        self.last_action = action
        self.last_log_prob = self._gaussian_log_prob(action)
        
        return action

    def update(self, observation, action, reward, next_observation, done=False):
        """
        Update the policy parameters (means) using a simple REINFORCE-style rule.
        Adjusts the means in the direction that maximizes the scalar reward.

        Args:
            observation (list): The state observed before choosing the action.
            action (list): The raw action vector that was chosen.
            reward (float): The scalar reward evaluated from the system's performance.
            next_observation (list): The resulting next state of the environment.
            done (bool): Whether the episode has concluded.
        """
        # Safety guard: Cannot update if memory buffers are empty or action was skipped
        if self.last_log_prob is None or action is None:
            return

        # Calculate the log-gradient for a Gaussian policy: d(log_prob)/d(mu) = (action - mu) / variance
        grads = [(a - mu) / (self.action_std ** 2) for a, mu in zip(action, self.means)]
        
        # Nudge the policy parameters (means) proportional to: learning_rate * reward * gradient
        for i, grad in enumerate(grads):
            self.means[i] += self.learning_rate * reward * grad

        # Advance state memory and flush step buffers
        self.last_observation = next_observation
        self.last_action = None
        self.last_log_prob = None

    def record_step(self, observation, raw_adjustments, adjustments, reward, death_count=0):
        """Record a single governor decision step for later inspection."""
        self.history.append({
            "observation": observation,
            "raw_adjustments": raw_adjustments,
            "adjustments": adjustments,
            "reward": reward,
            "death_count": death_count,
        })

    def reset(self):
        """Reset internal trajectory memories between individual simulation episodes."""
        self.last_observation = None
        self.last_action = None
        self.last_log_prob = None

    def _gaussian_log_prob(self, action):
        """
        Compute the log probability density of a chosen vector under the current policy.
        Used analytically to evaluate policy performance in standard policy gradient techniques.
        """
        variance = self.action_std ** 2
        
        # Evaluates the standard Gaussian log probability density function for each element
        log_probs = [-(0.5 * ((a - mu) ** 2) / variance + 0.5 * math.log(2 * math.pi * variance))
                     for a, mu in zip(action, self.means)]
                     
        return sum(log_probs)

    def _zero_sum_projection(self, raw_adjustments, alive_mask):
        """
        Project raw governor adjustments into a strictly zero-sum vector over alive agents only.
        This enforces a closed economy where Total Taxes Collected == Total Subsidies Paid Out.

        Args:
            raw_adjustments (list): Raw continuous values sampled from the policy.
            alive_mask (list of bool): True if agent is alive, False if dead.

        Returns:
            list: Mathematically balanced redistribution vector summing exactly to 0.0.
        """
        # Extract adjustments only for agents that are currently surviving
        alive_values = [u for u, alive in zip(raw_adjustments, alive_mask) if alive]
        
        # If no agents are alive, no economic activity can occur
        if not alive_values:
            return [0.0] * self.n_agents

        # Compute the mean of the raw adjustments among surviving agents
        mean_value = sum(alive_values) / len(alive_values)
        
        # Subtract the mean from alive agents to center the sum perfectly at 0.0.
        # Forced to a strict 0.0 if the agent is dead so they are excluded from the economy.
        return [u - mean_value if alive else 0.0
                for u, alive in zip(raw_adjustments, alive_mask)]

    def compute_governor_reward(self, final_rewards, death_count=0, death_penalty=None):
        """
        Compute the scalar reward feedback evaluated by the governor.
        Forces the governor to optimize for total system productivity while protecting lives.

        Args:
            final_rewards (list): Final economic payouts after redistribution.
            death_count (int): Number of agents that starved/died on this step.
            death_penalty (float, optional): Negative punishment magnitude incurred per agent death.
                If omitted, the governor's configured death_penalty is used.

        Returns:
            float: Total net reward evaluating governor policy utility.
        """
        if death_penalty is None:
            death_penalty = self.death_penalty

        # Base utility: Total gross resource extraction across the entire group
        total_reward = sum(final_rewards)
        
        # Penalize severely if the governor's actions allowed agents to cross the death threshold
        return total_reward - death_penalty * death_count