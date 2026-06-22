from unittest import result

from multi_agent_bandits.core import agent
from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import custom_100_20_10_share, linear_share


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
		self.initial_wealth = initial_wealth

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
		
		# Finally, call agent.update() for every agent so alive agents learn from their final rewards
		for agent_idx, agent in enumerate(agents):
            # Pass the final post-tax/post-subsidy reward to the agent
			agent.update(final_rewards[agent_idx]) 

		return choices, final_rewards


class DepletingCommonsEnvironment(Environment):
    """
    Dynamic Depleting Commons Environment.
    
    Behavior:
    - Default to 3 arms with means [10, 2, 1] unless `arms` is provided.
    - Uses the imported `linear_share` function to cleanly split rewards on collisions.
    - Arms have an environmental health factor that degrades under heavy congestion
      (>1 agent) and slowly regenerates when left alone or used sustainably.
    - Tracks per-agent wealth and marks agents as dead when wealth < death_threshold.
    """

    def __init__(self, n_agents=3, arms=None, death_threshold=0.0, initial_wealth=0.0, step_cost=0.0, governor=None, delta_drain=0.15, gamma_regen=0.05):
        # Build default arms if none provided
        if arms is None:
            arms = [
			# ==== Tier 1: The "Mega-Yield" Gems (High Value, High Competition) ====
			Arm(mean=10.0, sd=1.0),  # Arm 0: The original goldmine
			Arm(mean=9.0,  sd=1.0),  # Arm 1
			Arm(mean=8.2,  sd=1.0),  # Arm 2
			Arm(mean=7.5,  sd=1.0),  # Arm 3
			Arm(mean=6.8,  sd=1.0),  # Arm 4

			# ==== Tier 2: The "Industrial Middle Class" (Reliable, Moderate Yield) ====
			Arm(mean=5.5,  sd=1.0),  # Arm 5
			Arm(mean=5.0,  sd=1.0),  # Arm 6
			Arm(mean=4.6,  sd=1.0),  # Arm 7
			Arm(mean=4.2,  sd=1.0),  # Arm 8
			Arm(mean=3.9,  sd=1.0),  # Arm 9
			Arm(mean=3.6,  sd=1.0),  # Arm 10
			Arm(mean=3.3,  sd=1.0),  # Arm 11
			Arm(mean=3.0,  sd=1.0),  # Arm 12
			Arm(mean=2.8,  sd=1.0),  # Arm 13
			Arm(mean=2.6,  sd=1.0),  # Arm 14

			# ==== Tier 3: The "Subsistence/Scarcity" Tail (Low Yield, Safety Nets) ====
			Arm(mean=2.0,  sd=1.0),  # Arm 15 (Your original Arm 1)
			Arm(mean=1.9,  sd=1.0),  # Arm 16
			Arm(mean=1.8,  sd=1.0),  # Arm 17
			Arm(mean=1.7,  sd=1.0),  # Arm 18
			Arm(mean=1.6,  sd=1.0),  # Arm 19
			Arm(mean=1.5,  sd=1.0),  # Arm 20
			Arm(mean=1.4,  sd=1.0),  # Arm 21
			Arm(mean=1.3,  sd=1.0),  # Arm 22
			Arm(mean=1.2,  sd=1.0),  # Arm 23
			Arm(mean=1.1,  sd=1.0),  # Arm 24
			Arm(mean=1.0,  sd=1.0),  # Arm 25 (Your original Arm 2)
			Arm(mean=0.9,  sd=1.0),  # Arm 26
			Arm(mean=0.8,  sd=1.0),  # Arm 27
			Arm(mean=0.7,  sd=1.0),  # Arm 28
			Arm(mean=0.5,  sd=1.0),  # Arm 29: The absolute baseline desert
		]

        # Initialize base environment using the imported linear_share function
        super().__init__(n_agents=n_agents, arms=arms, collision_policy=linear_share)

        # Ecological Parameters
        self.arm_health = [1.0] * self.n_arms    # All arms start at 100% capacity (1.0)
        self.delta_drain = delta_drain          # Health lost per step when crowded (>1 agent)
        self.gamma_regen = gamma_regen          # Health recovered per step when rested (<=1 agent)
        self.environmental_health_history = []  # Log history for thesis plotting

        # Standard parameters and runtime state
        self.death_threshold = death_threshold
        self.step_cost = step_cost
        self.agent_wealths = [initial_wealth] * n_agents
        self.is_alive = [True] * n_agents
        self.wealth_history = []
        self.death_steps = [None] * n_agents
        self.current_step = 0
        self.governor = governor
        self.governor_reward_history = []
        self.initial_wealth = initial_wealth

    def _build_governor_observation(self, choices):
        choice_obs = [choice if choice is not None else -1 for choice in choices]
        # Crucial for the AI Governor: We append the arm health states to the observation vector
        # so an intelligent agent can see the resource depleting in real-time!
        return choice_obs + list(self.agent_wealths) + list(self.arm_health)

    def step(self, agents):
        #[Phase 1: Bandit Choices]
        choices = [None] * len(agents)
        collisions = {}

        for agent_idx, agent in enumerate(agents):
            if not self.is_alive[agent_idx]:
                continue

            chosen_arm = agent.choose_arm()
            choices[agent_idx] = chosen_arm
            collisions.setdefault(chosen_arm, []).append(agent_idx)

        #[Phase 2: Harvest & Collision via linear_share + Health Scaling]
        raw_rewards = [0.0] * len(agents)

        # 1. Harvest rewards based on CURRENT step's environmental health
        for arm_idx, agent_ids in collisions.items():
            n_agents_on_arm = len(agent_ids)
            base_reward = self.sample_reward(arm_idx)
            
            # Scale the total reward pool by the current health of this specific arm
            total_available_reward = base_reward * self.arm_health[arm_idx]

            # Use the cleanly imported linear_share function
            shares = linear_share(total_available_reward, n_agents_on_arm)
            
            for share, a_id in zip(shares, agent_ids):
                raw_rewards[a_id] = share

        # 2. State Transition: Update environmental health for the NEXT timestep
        for arm_idx in range(self.n_arms):
            agents_on_arm = len(collisions.get(arm_idx, []))

            if agents_on_arm > 1:
                # Scale drainage by how many EXTRA agents are crowding the arm
				# 2 agents = 1 * delta_drain
				# 10 agents = 9 * delta_drain
                excess_agents = agents_on_arm - 1
                total_drain = excess_agents * self.delta_drain
                
                self.arm_health[arm_idx] = max(0.1, self.arm_health[arm_idx] - total_drain)
            else:
                # Sustained (1 agent) or Rested (0 agents): Resource naturally recovers up to 1.0
                self.arm_health[arm_idx] = min(1.0, self.arm_health[arm_idx] + self.gamma_regen)

        # Log environmental state snapshot
        self.environmental_health_history.append(list(self.arm_health))
        final_rewards = list(raw_rewards)
        
        #[Phase 3: Governor Tax/Subsidy]
        adjustments = [0.0] * len(agents)

        if self.governor and any(self.is_alive):
            observation = self._build_governor_observation(choices)
            # Allow governors to return either (action_idx, adjustments)
            # or just adjustments. Be robust to both interfaces.
            result = self.governor.choose_action(
                observation=observation,
                choices=choices,
                wealths=list(self.agent_wealths),
                raw_rewards=list(raw_rewards),
                alive_mask=list(self.is_alive),
            )

            action_idx = None
            adjustments = None
            if isinstance(result, (tuple, list)) and len(result) == 2:
                maybe_idx, maybe_adj = result
                try:
                    action_idx = int(maybe_idx)
                    adjustments = maybe_adj
                except Exception:
                    adjustments = result
            else:
                adjustments = result

            # Optionally record strategy performance if the governor provided an index
            if action_idx is not None and hasattr(self.governor, "update_strategy_performance"):
                try:
                    self.governor.update_strategy_performance(
                        action_idx=action_idx,
                        reward=sum(adjustments) if adjustments is not None else 0.0,
                    )
                except Exception:
                    pass

            # Apply computed adjustments to alive agents
            for agent_idx, adjustment in enumerate(adjustments):
                if self.is_alive[agent_idx]:
                    final_rewards[agent_idx] = raw_rewards[agent_idx] + adjustment
                else:
                    final_rewards[agent_idx] = 0.0
        else:
            final_rewards = list(raw_rewards)

        #[Phase 4: Economic Accounting]
        self.current_step += 1
        death_count = 0

        for agent_idx in range(len(agents)):
            if not self.is_alive[agent_idx]:
                final_rewards[agent_idx] = 0.0
                continue

            self.agent_wealths[agent_idx] += final_rewards[agent_idx] - self.step_cost

            if self.agent_wealths[agent_idx] < self.death_threshold:
                self.is_alive[agent_idx] = False
                self.death_steps[agent_idx] = self.current_step
                final_rewards[agent_idx] = 0.0
                death_count += 1

        self.wealth_history.append(list(self.agent_wealths))

       # --- NEW PPO ENVIRONMENTAL RECORDING HOOK ---
        if self.governor:
            current_arm_healths = list(self.arm_health) if hasattr(self, 'arm_health') else [1.0] * self.n_arms
            current_wealths = list(self.agent_wealths)
            
            # Step 1: Record data step-by-step into the PPO Trajectory Buffer
            if hasattr(self.governor, "record_step_data"):
                self.governor.record_step_data(
                    gross_rewards=final_rewards,
                    arm_healths=current_arm_healths,
                    wealths=current_wealths,
                    death_count=death_count
                )
        # ---------------------------------------------
        
        # Trigger reinforcement learning updates for agents
        for agent_idx, agent in enumerate(agents):
            agent.update(final_rewards[agent_idx])

        return choices, final_rewards