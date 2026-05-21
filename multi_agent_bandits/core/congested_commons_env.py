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

	def __init__(self, n_agents=3, arms=None, death_threshold=0.0, initial_wealth=0.0, step_cost=0.0, governor=None):
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
		self.step_cost = step_cost
		self.agent_wealths = [initial_wealth] * n_agents  # accumulated wealth per agent
		self.is_alive = [True] * n_agents  # alive/dead flags per agent
		self.wealth_history = []  # snapshot of wealth after each timestep
		self.death_steps = [None] * n_agents  # timestep when agent first died
		self.current_step = 0
		self.governor = governor
		self.governor_reward_history = []

	def _build_governor_observation(self, choices):
		"""Build a fixed-size observation for the governor.

		Choices are encoded as integers with -1 for dead / inactive agents.
		Wealth values are appended directly so the governor sees the current state.
		"""
		choice_obs = [choice if choice is not None else -1 for choice in choices]
		return choice_obs + list(self.agent_wealths)

	def step(self, agents):
		"""
		Run a single timestep for the environment:
		[Phase 1: Bandit Choices] ──► [Phase 2: Harvest & Collision] ──► [Phase 3: Governor Tax/Subsidy] ──► [Phase 4: Economic Accounting]

		Key points:
		- Dead agents do not make choices and do not contribute to collisions.
		- Uses `custom_100_20_10_share` directly to split raw rewards among
		  the agents that pulled the same arm.
		- Updates agent wealths and applies the death threshold.
		- Calls `agent.update(reward)` for every agent so alive agents keep
		  receiving their learning signal (dead agents receive 0 and
		  are still passed through `update` with 0).
		"""

		#[Phase 1: Bandit Choices]
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

		#[Phase 2: Harvest & Collision]
		
		# Prepare reward container (default 0 for everyone)
		raw_rewards = [0.0] * len(agents)

		# For each arm that was pulled by one or more alive agents,
		# sample a raw reward and split it using the imported congestion policy.
		for arm_idx, agent_ids in collisions.items():
			raw_reward = self.sample_reward(arm_idx)

			# Use the exact math already implemented in reward_sharing.py
			shares = custom_100_20_10_share(raw_reward, len(agent_ids))

			# Distribute computed shares back to the correct agents
			for share, a_id in zip(shares, agent_ids):
				raw_rewards[a_id] = share

		final_rewards = list(raw_rewards)
		
		#[Phase 3: Governor Tax/Subsidy]

		adjustments = [0.0] * len(agents)
			
		if self.governor and any(self.is_alive):
			# The environment builds a clean observation vector for the brain
			observation = self._build_governor_observation(choices)
			
			# THE INTERFACE: We pass absolutely EVERYTHING available into the governor
			adjustments = self.governor.choose_action(
				observation=observation,
				choices=choices,
				wealths=list(self.agent_wealths),
				raw_rewards=list(raw_rewards),
				alive_mask=list(self.is_alive)
			)
			
			# Apply adjustments to calculate final scores
			for agent_idx, adjustment in enumerate(adjustments):
				if self.is_alive[agent_idx]:
					final_rewards[agent_idx] = raw_rewards[agent_idx] + adjustment
				else:
					final_rewards[agent_idx] = 0.0
		else:
			final_rewards = list(raw_rewards)

		#[Phase 4: Economic Accounting]

		# Increment the internal timestep counter for death tracking
		self.current_step += 1
		# Track deaths that occur during this timestep
		death_count = 0

		# Keep a snapshot of final rewards *before* any potential mid-loop death reset 
        # to ensure the governor and training logs receive clean, un-bugged reward data
		payouts_for_training = list(final_rewards)

		# Update wealths and apply death checks for alive agents
		for agent_idx in range(len(agents)):
			if not self.is_alive[agent_idx]:
				# Ensure dead agents have zero reward and do not pay costs
				final_rewards[agent_idx] = 0.0
				continue

			# Apply reward and scarcity step cost for alive agents
			self.agent_wealths[agent_idx] += final_rewards[agent_idx] - self.step_cost

			# If wealth falls below threshold, agent dies and no longer
			# participates from the next timestep onwards.
			if self.agent_wealths[agent_idx] < self.death_threshold:
				self.is_alive[agent_idx] = False
				self.death_steps[agent_idx] = self.current_step
				final_rewards[agent_idx] = 0.0
				death_count += 1

		# Record the wealth snapshot after this timestep has finished
		self.wealth_history.append(list(self.agent_wealths))

		# Initialize a blank variable to calculate the governor's grade later
		governor_reward = 0.0

		# Calculate governor reward after redistribution and death penalties
		if self.governor:
			governor_reward = self.governor.compute_governor_reward(final_rewards, death_count)
			self.governor_reward_history.append(governor_reward)
			if hasattr(self.governor, "record_step"):
				self.governor.record_step(
                    observation, 
                    adjustments, 
                    adjustments, 
                    governor_reward, 
                    death_count=death_count
                )
		# Update governor training signal if available
		if self.governor and self.governor.last_action is not None:
			next_observation = self._build_governor_observation(choices)
			self.governor.update(
                observation, 
                adjustments, 
                governor_reward, 
                next_observation, 
                done=False
            )

		return choices, final_rewards

