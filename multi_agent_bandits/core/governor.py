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

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Processes the environment state and returns a finalized, budget-balanced 
        zero-sum tax/subsidy adjustment vector for the living agents.

        Args:
            observation (list): Processed state vector (choices + wealth levels).
            choices (list): Raw choices made by the bandits this turn.
            wealths (list): Current financial ledger levels for each agent.
            raw_rewards (list): Freshly harvested reward payouts before taxes.
            alive_mask (list of bool): Boolean flags indicating who is alive.

        Returns:
            list: Finished zero-sum continuous adjustment vector of size n_agents.
        """
        self.last_observation = observation
        
        # 1. Sample raw adjustments from normal distributions centered around self.means
        raw_action = [random.gauss(mu, self.action_std) for mu in self.means]
        
        # 2. Project them to a balanced closed-economy vector right here inside the brain
        balanced_action = self._zero_sum_projection(raw_action, alive_mask)
        
        # 3. Cache the actions and policy log probabilities for the gradient calculation later
        self.last_action = balanced_action
        self.last_log_prob = self._gaussian_log_prob(balanced_action)
        
        return balanced_action

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
        """
        alive_values = []
        for u, alive in zip(raw_adjustments, alive_mask):
            if alive:
                alive_values.append(u)
        
        # If no agents are alive, no economic activity can occur
        if not alive_values:
            return [0.0] * self.n_agents

        # Compute the mean of the raw adjustments among surviving agents
        mean_value = sum(alive_values) / len(alive_values)
        
        # Subtract the mean from alive agents to center the sum perfectly at 0.0.
        # Forced to a standard 0.0 if the agent is dead so they are excluded.
        balanced_adjustments = []
        for u, alive in zip(raw_adjustments, alive_mask):
            if alive:
                balanced_adjustments.append(u - mean_value)
            else:
                balanced_adjustments.append(0.0)
                
        return balanced_adjustments

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
    




class CommunistGovernor:
    """
    An omniscient, non-learning baseline governor.
    
    This governor automatically calculates a perfect egalitarian split 
    of the gathered resources. It intercepts the raw rewards, sums them up, 
    and issues taxes and subsidies so that every single living agent receives 
    an identical final payout.
    """

    def __init__(self, n_agents=3):
        """
        Initialize the Communist Governor baseline.
        """
        self.n_agents = n_agents
        
        # This property acts as a marker for the environment file.
        # Setting it to None tells the environment that this governor 
        # doesn't use policy gradient adjustments and should bypass .update()
        self.last_action = None 

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Calculates the tax/subsidy adjustment matrix needed to equalize wealth gains.

        Args:
            observation (list): Processed state vector (ignored by this governor).
            choices (list): Raw choices made by the bandits this turn (ignored).
            wealths (list): Current financial ledger levels for each agent (ignored).
            raw_rewards (list): Freshly harvested reward payouts before taxes.
            alive_mask (list of bool): Boolean flags indicating who is alive.

        Returns:
            list: Finished zero-sum continuous adjustment vector of size n_agents.
        """
        # 1. Count how many agents are currently alive
        num_alive = 0
        for alive in alive_mask:
            if alive:
                num_alive += 1

        # If everyone has starved to death, no economy happens
        if num_alive == 0:
            return [0.0] * self.n_agents

        # 2. Sum up all the raw wealth harvested by the alive agents this round
        total_harvest = 0.0
        for idx, reward in enumerate(raw_rewards):
            if alive_mask[idx]:
                total_harvest += reward

        # 3. Calculate the equal split target value
        equal_share = total_harvest / num_alive

        # 4. Generate the adjustments vector: (Equal Share Target - Their Raw Harvest)
        adjustments = []
        for idx, reward in enumerate(raw_rewards):
            if alive_mask[idx]:
                # If an agent harvested more than the equal share, this becomes negative (tax).
                # If they harvested less, it becomes positive (subsidy).
                tax_or_subsidy = equal_share - reward
                adjustments.append(tax_or_subsidy)
            else:
                # Dead agents are completely isolated from the redistributive pool
                adjustments.append(0.0)

        return adjustments

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates performance to keep the historical metrics tracking unbroken.
        """
        # Total productivity across the entire pool
        return sum(final_rewards)
    

class PigouvianGovernor:
    """
    A rule-based baseline governor that implements a targeted Pigouvian tax.
    
    Instead of pooling all wealth equally, it calculates real-time crowding taxes 
    on over-exploited arms to punish greedy actions. It then funnels those taxes 
    exclusively as survival subsidies to agents close to bankruptcy.
    """

    def __init__(self, n_agents=30, delta_drain=0.15, survival_threshold=10.0):
        """
        Initialize the Pigouvian Governor.

        Args:
            n_agents (int): Total number of agents in the system.
            delta_drain (float): The environmental damage multiplier from your environment file.
            survival_threshold (float): Wealth level below which an agent qualifies for a subsidy.
        """
        self.n_agents = n_agents
        self.delta_drain = delta_drain
        self.survival_threshold = survival_threshold
        
        # This property acts as a marker for the environment runner.
        # Setting it to None tells the environment to skip the policy gradient .update()
        self.last_action = None

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Calculates the targeted crowding tax matrix and safely redistributes 
        collected fines to starving agents.

        Args:
            observation (list): Processed state vector (ignored by this governor).
            choices (list): Raw choices made by the bandits this turn.
            wealths (list): Current financial ledger levels for each agent.
            raw_rewards (list): Freshly harvested reward payouts before taxes (ignored).
            alive_mask (list of bool): Boolean flags indicating who is alive.

        Returns:
            list: Finished zero-sum continuous adjustment vector of size n_agents.
        """
        # 1. Count how many agents chose each arm this turn
        arm_counts = {}
        for idx, choice in enumerate(choices):
            if alive_mask[idx]:
                arm_counts[choice] = arm_counts.get(choice, 0) + 1

        # 2. Charge a targeted crowding tax to agents occupying overcrowded arms
        total_tax_collected = 0.0
        individual_taxes = [0.0] * self.n_agents

        for idx, choice in enumerate(choices):
            if alive_mask[idx]:
                agents_on_arm = arm_counts.get(choice, 0)
                
                if agents_on_arm > 1:
                    # Excess agents past the sustainable 1-agent limit
                    excess_agents = agents_on_arm - 1
                    
                    # Tax matches the exact dynamic ecological damage caused to the arm
                    tax_amount = excess_agents * self.delta_drain
                    
                    individual_taxes[idx] = tax_amount
                    total_tax_collected += tax_amount

        # 3. Identify struggling agents that qualify for a survival subsidy
        struggling_agents = []
        for idx, wealth in enumerate(wealths):
            if alive_mask[idx] and wealth < self.survival_threshold:
                struggling_agents.append(idx)

        # 4. Redistribute collected taxes to ensure a closed, zero-sum economy
        adjustments = [0.0] * self.n_agents

        if total_tax_collected > 0.0:
            if struggling_agents:
                # Target Option A: Distribute equally among agents near the death threshold
                subsidy_per_strugler = total_tax_collected / len(struggling_agents)
                for idx in struggling_agents:
                    adjustments[idx] = subsidy_per_strugler
            else:
                # Target Option B: If everyone is wealthy, give it to agents on underused arms
                underused_agents = []
                for idx, choice in enumerate(choices):
                    if alive_mask[idx] and arm_counts.get(choice, 0) == 1:
                        underused_agents.append(idx)
                        
                if underused_agents:
                    subsidy_per_underused = total_tax_collected / len(underused_agents)
                    for idx in underused_agents:
                        adjustments[idx] = subsidy_per_underused
                else:
                    # Fallback Option C: If everyone is on crowded arms, give a flat refund
                    num_alive = sum(alive_mask)
                    subsidy_flat = total_tax_collected / num_alive
                    for idx, alive in enumerate(alive_mask):
                        if alive:
                            adjustments[idx] = subsidy_flat

        # 5. Combine the negative taxes and positive subsidies into the final adjustment vector
        final_adjustments = []
        for idx, alive in enumerate(alive_mask):
            if alive:
                # Adjustment = Subsidy received minus Tax paid
                net_change = adjustments[idx] - individual_taxes[idx]
                final_adjustments.append(net_change)
            else:
                # Dead agents are completely cut off from the economic loop
                final_adjustments.append(0.0)

        return final_adjustments

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates performance to keep the historical metrics tracking unbroken.
        """
        # Total productivity across the entire pool
        return sum(final_rewards)
    

class SocialistGovernor:
    """
    A mixed-incentive baseline governor.
    
    This governor implements an adjustable split economy:
    - (1 - tax_rate)% of each agent's raw harvest is kept purely as private property.
    - tax_rate% of each agent's raw harvest is collectivized into a central pool.
    
    The central pool is summed up and distributed perfectly equally among 
    all living agents.
    """

    def __init__(self, n_agents=3, tax_rate=0.5):
        """
        Initialize the Socialist Governor baseline.

        Args:
            n_agents (int): Number of agents in the simulation.
            tax_rate (float): The percentage of raw rewards to collectivize (between 0.0 and 1.0).
        """
        self.n_agents = n_agents
        self.tax_rate = tax_rate
        
        # Setting this to None tells the environment runner that this governor 
        # is a heuristic baseline and does not require policy gradient .update() hooks
        self.last_action = None 

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Calculates the tax/subsidy adjustments required to enforce the split economy.

        Args:
            observation (list): Processed state vector (ignored by this governor).
            choices (list): Raw choices made by the bandits this turn (ignored).
            wealths (list): Current financial ledger levels for each agent (ignored).
            raw_rewards (list): Freshly harvested reward payouts before taxes.
            alive_mask (list of bool): Boolean flags indicating who is alive.

        Returns:
            list: Finished zero-sum continuous adjustment vector of size n_agents.
        """
        # 1. Count how many agents are currently alive
        num_alive = sum(1 for alive in alive_mask if alive)

        # If everyone has starved to death, no economic activity can occur
        if num_alive == 0:
            return [0.0] * self.n_agents

        # 2. Collectivize the configured tax percentage of the raw wealth harvested by surviving agents
        total_socialized_pool = 0.0
        for idx, reward in enumerate(raw_rewards):
            if alive_mask[idx]:
                total_socialized_pool += self.tax_rate * reward

        # 3. Calculate the equal social dividend payout per living agent
        social_dividend = total_socialized_pool / num_alive

        # 4. Generate the zero-sum adjustments vector.
        # Target Final Payout = ((1 - tax_rate) * Raw Reward) + Social Dividend
        # Adjustment = Target Final Payout - Raw Reward
        # Simplified: Adjustment = Social Dividend - (tax_rate * Raw Reward)
        adjustments = []
        for idx, reward in enumerate(raw_rewards):
            if alive_mask[idx]:
                # Agents who produce above the community average face a net tax (negative adjustment).
                # Agents who produce below the community average receive a net subsidy (positive adjustment).
                tax_or_subsidy = social_dividend - (self.tax_rate * reward)
                adjustments.append(tax_or_subsidy)
            else:
                # Dead agents cannot pay taxes or receive social dividends
                adjustments.append(0.0)

        return adjustments

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates performance to keep historical experiment metric logs unbroken.
        """
        # Evaluates total macroeconomic productivity across the entire group
        return sum(final_rewards)
    

class FreeMarketGovernor:
    """
    A laissez-faire governor baseline.
    Enforces a 100% private property economy by applying zero adjustments.
    """
    def __init__(self, n_agents=30):
        self.n_agents = n_agents
        self.last_action = None

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        # Simply return zero adjustments for every agent
        return [0.0] * self.n_agents

    def compute_governor_reward(self, final_rewards, death_count=0):
        return sum(final_rewards)

class SafetyNetGovernor:
    """
    A minimal-intervention emergency governor.
    
    - Under normal conditions, it leaves raw rewards untouched (0% tax rate).
    - If any agent's wealth drops below a specified safety buffer, it steps in.
    - It calculates exactly how much wealth is missing to guarantee survival this step.
    - It collects this exact amount by evenly taxing all agents who are safely above 
      the danger line, ensuring a zero-sum, redistribution safety net.
    """

    def __init__(self, n_agents=30, safety_buffer=15.0, step_cost=3.0):
        """
        Initialize the Safety Net Governor.

        Args:
            n_agents (int): Total number of agents.
            safety_buffer (float): The wealth threshold *before* step_cost below which 
                                  emergency intervention is triggered.
            step_cost (float): Environmental cost subtracted per turn (to accurately forecast death).
        """
        self.n_agents = n_agents
        self.safety_buffer = safety_buffer
        self.step_cost = step_cost
        self.last_action = None

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Intervenes only to prevent death, calculating precise micro-taxes.
        """
        num_alive = sum(1 for alive in alive_mask if alive)
        if num_alive <= 1:
            # If 0 or 1 agent is left, redistribution is impossible or meaningless
            return [0.0] * self.n_agents

        # Initialize adjustments at zero (default: free-market / no intervention)
        adjustments = [0.0] * self.n_agents

        # 1. Forecast next-step wealth to catch dying agents
        # Expected Wealth = Current Wealth + Raw Reward - Step Cost
        expected_wealths = []
        for idx in range(self.n_agents):
            if alive_mask[idx]:
                expected_wealths.append(wealths[idx] + raw_rewards[idx] - self.step_cost)
            else:
                expected_wealths.append(-float('inf'))

        # 2. Identify who needs a rescue and calculate total bailout fund required
        total_bailout_needed = 0.0
        rescue_targets = []

        for idx, exp_wealth in enumerate(expected_wealths):
            if alive_mask[idx] and exp_wealth < self.safety_buffer:
                needed_subsidy = self.safety_buffer - exp_wealth
                total_bailout_needed += needed_subsidy
                rescue_targets.append((idx, needed_subsidy))

        # If no one is in danger, exit early with zero taxes!
        if total_bailout_needed == 0:
            return adjustments

        # 3. Identify who is healthy enough to pay taxes
        # Eligible taxpayers must be alive and safely above the safety buffer
        taxpayers = [idx for idx, exp_wealth in enumerate(expected_wealths) 
                     if alive_mask[idx] and exp_wealth > self.safety_buffer]

        # Edge case: If the whole society is crumbling and nobody is healthy enough to tax,
        # we try to distribute the tax burden evenly among all living agents instead.
        if not taxpayers:
            taxpayers = [idx for idx, alive in enumerate(alive_mask) if alive]

        # 4. Apportion the tax evenly among taxpayers and distribute to the vulnerable
        tax_per_taxpayer = total_bailout_needed / len(taxpayers)

        # Apply negative adjustments (taxes)
        for idx in taxpayers:
            adjustments[idx] -= tax_per_taxpayer

        # Apply positive adjustments (subsidies)
        for idx, subsidy in rescue_targets:
            adjustments[idx] += subsidy

        return adjustments

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Macroeconomic evaluation tracking system productivity.
        """
        return sum(final_rewards)
    

class DynamicTaxingGovernor:
    """
    Discrete-action meta-governor using a Softmax policy gradient skeleton.
    
    Instead of directly modifying individual wealth vectors, this AI selects 
    an entire sub-governor strategy to handle economic routing each round.
    
    Action Map:
      0: CommunistGovernor()
      1: SocialistGovernor(tax_rate=0.3)
      2: PigouvianGovernor(delta_drain=0.15)
      3: FreeMarketGovernor()
    """

    def __init__(self, n_agents=30, learning_rate=0.05, death_penalty=100.0, seed=None):
        """
        Initialize the Dynamic Macro-Governor AI.
        """
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.death_penalty = death_penalty
        
        # 1. Instantiate the available sub-governors as the action space
        self.strategies = [
            CommunistGovernor(n_agents=n_agents),
            SocialistGovernor(n_agents=n_agents, tax_rate=0.3),
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            FreeMarketGovernor(n_agents=n_agents)
        ]
        self.n_actions = len(self.strategies)
        
        # 2. Policy parameters: Numerical preferences (logits) for each strategy.
        # Initialized to 0.0 so all policies have an equal probability at the start.
        self.logits = [0.0] * self.n_actions
        
        self.last_action = None

        # Memory storage buffers for discrete policy gradient step tracking
        self.last_action_idx = None
        self.last_action_probs = None
        self.history = []

        if seed is not None:
            random.seed(seed)

    def _get_action_probabilities(self):
        """
        Compute the Softmax probability distribution over the available macro-policies.
        """
        # Stability fix: Subtract max logit to avoid numeric overflow explosions
        max_logit = max(self.logits)
        exp_logits = [math.exp(l - max_logit) for l in self.logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Selects a macro-economic strategy using a Softmax distribution, and then 
        delegates the step calculations to that strategy.
        """
        # 1. Calculate current probability distribution across your strategies
        probs = self._get_action_probabilities()
        self.last_action_probs = probs
        
        # 2. Sample a macro-policy choice based on those probabilities
        r = random.random()
        cumulative_prob = 0.0
        chosen_idx = 0
        for idx, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                chosen_idx = idx
                break
        
        self.last_action_idx = chosen_idx
        
        # 3. Delegate execution directly to the chosen sub-governor strategy
        selected_strategy = self.strategies[chosen_idx]
        adjustments = selected_strategy.choose_action(
            observation, choices, wealths, raw_rewards, alive_mask
        )
        
        # --- CRITICAL FIX: Give the environment loop a non-None value to trigger update() ---
        self.last_action = adjustments 
        # -------------------------------------------------------------------------------------
        
        return adjustments

    def update(self, observation, action, reward, next_observation, done=False):
        """
        Update strategy preferences (logits) using the discrete REINFORCE update rule.
        """
        if self.last_action_idx is None or self.last_action_probs is None:
            return

        # STABILITY SCALING: Scale down large reward magnitudes (e.g., 50,000+) 
        # so gradients don't break the exponential math in Softmax
        scaled_reward = reward / 10000.0 

        # Nudge our strategic preferences up or down based on performance
        for idx in range(self.n_actions):
            if idx == self.last_action_idx:
                grad = 1.0 - self.last_action_probs[idx]
            else:
                grad = -self.last_action_probs[idx]
            
            # logit = logit + alpha * Scaled_Reward * Gradient
            self.logits[idx] += self.learning_rate * scaled_reward * grad

        # Reset tracking buffers and clean up environment flag
        self.last_action_idx = None
        self.last_action_probs = None
        self.last_action = None  # Reset back to None for the next step cycle

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates system performance. Combines raw resource extraction 
        with a massive structural penalty for enabling bankruptcies.
        """
        total_reward = sum(final_rewards)
        return total_reward - (self.death_penalty * death_count)

    def get_policy_distribution_string(self):
        """Helper function to print what your agent is thinking during batch runs."""
        probs = self._get_action_probabilities()
        names = ["Communist", "Socialist", "Pigouvian", "FreeMarket"]
        return ", ".join([f"{name}: {p*100:.1f}%" for name, p in zip(names, probs)])