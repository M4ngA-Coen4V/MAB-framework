from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import custom_100_20_10_share


class CongestedCommonsEnvironment(Environment):
	"""
	Congested commons environment subclass.

	Behavior:
	- Default to 3 arms with means [10, 2, 1] unless `arms` is provided.
	- Uses the existing `custom_100_20_10_share` function from
	  `multi_agent_bandits.core.reward_sharing` to split rewards on collisions.
	- Tracks per-agent wealth and marks agents as dead when wealth < death_threshold.
	- Dead agents do not choose arms, do not cause collisions, and receive 0 reward.
	"""

	def __init__(self, n_agents=3, arms=None, death_threshold=0.0, initial_wealth=0.0):
		# Build default arms if none provided
		if arms is None:
			arms = [
				Arm(mean=10.0, sd=1.0),
				Arm(mean=2.0, sd=1.0),
				Arm(mean=1.0, sd=1.0),
			]

		# Initialize the base Environment. We still pass the congestion
		# function for compatibility, but the overridden `step` will call
		# `custom_100_20_10_share` directly as requested.
		super().__init__(n_agents=n_agents, arms=arms, collision_policy=custom_100_20_10_share)

		# Parameters and runtime state
		self.death_threshold = death_threshold
		self.agent_wealths = [initial_wealth] * n_agents  # accumulated wealth per agent
		self.is_alive = [True] * n_agents  # alive/dead flags per agent

	def step(self, agents):
		"""
		Run a single timestep for the environment.

		Key points:
		- Dead agents do not make choices and do not contribute to collisions.
		- Uses `custom_100_20_10_share` directly to split raw rewards among
		  the agents that pulled the same arm.
		- Updates agent wealths and applies the death threshold.
		- Calls `agent.update(reward)` for every agent so alive agents keep
		  receiving their learning signal (dead agents receive 0 and
		  are still passed through `update` with 0).
		"""

		# Record choices; dead agents remain `None` here.
		choices = [None] * len(agents)

		# Build collision groups only from alive agents' choices
		collisions = {}
		for agent_idx, agent in enumerate(agents):
			if not self.is_alive[agent_idx]:
				# Skip dead agents: they neither choose nor cause collisions
				continue

			# Agent selects an arm
			chosen_arm = agent.choose_arm()
			choices[agent_idx] = chosen_arm
			collisions.setdefault(chosen_arm, []).append(agent_idx)

		# Prepare reward container (default 0 for everyone)
		rewards = [0.0] * len(agents)

		# For each arm that was pulled by one or more alive agents,
		# sample a raw reward and split it using the imported congestion policy.
		for arm_idx, agent_ids in collisions.items():
			raw_reward = self.sample_reward(arm_idx)

			# Use the exact math already implemented in reward_sharing.py
			shares = custom_100_20_10_share(raw_reward, len(agent_ids))

			# Distribute computed shares back to the correct agents
			for share, a_id in zip(shares, agent_ids):
				rewards[a_id] = share

		# Update wealths and apply death checks for alive agents
		for agent_idx in range(len(agents)):
			if not self.is_alive[agent_idx]:
				# Ensure dead agents have zero reward
				rewards[agent_idx] = 0.0
				continue

			# Accumulate wealth for alive agents
			self.agent_wealths[agent_idx] += rewards[agent_idx]

			# If wealth falls below threshold, agent dies and no longer
			# participates from the next timestep onwards. We also zero
			# their reward immediately to reflect death.
			if self.agent_wealths[agent_idx] < self.death_threshold:
				self.is_alive[agent_idx] = False
				rewards[agent_idx] = 0.0

		# Ensure each agent receives an update call (alive or dead)
		for agent_idx, agent in enumerate(agents):
			agent.update(rewards[agent_idx])

		return choices, rewards

