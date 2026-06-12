import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



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
    
class ProgressiveTaxGovernor:
    '''
    Takes a set tax rate of the top 50% of earners and redistributes it to the bottom 50% of earners.
    '''
    def __init__(self, n_agents=30, tax_rate=0.4):
        self.n_agents = n_agents
        self.tax_rate = tax_rate

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        alive_indices = [i for i, alive in enumerate(alive_mask) if alive]
        if not alive_indices: return [0.0] * self.n_agents
        
        # Identify top/bottom performers based on harvest
        sorted_indices = sorted(alive_indices, key=lambda i: raw_rewards[i])
        num_bottom = len(alive_indices) // 2
        
        tax_pool = 0.0
        adjustments = [0.0] * self.n_agents
        
        # Collect tax from top 50%
        for i in sorted_indices[num_bottom:]:
            tax = raw_rewards[i] * self.tax_rate
            adjustments[i] -= tax
            tax_pool += tax
            
        # Redistribute to bottom 50%
        subsidy = tax_pool / max(1, num_bottom)
        for i in sorted_indices[:num_bottom]:
            adjustments[i] += subsidy
            
        return adjustments
    
    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates performance to keep the historical metrics tracking unbroken.
        """
        # Total productivity across the entire pool
        return sum(final_rewards)

class SurvivalTargetedGovernor:
    '''
    Takes a changing amount of tax from all agents to make sure every agent gets atleast the survival_cost, 
    then redistributes it to the agents that put in most effort.
    '''
    def __init__(self, n_agents=30, survival_cost=3.0):
        self.n_agents = n_agents
        self.survival_cost = survival_cost

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        alive_indices = [i for i, alive in enumerate(alive_mask) if alive]
        num_alive = len(alive_indices)
        if num_alive == 0: return [0.0] * self.n_agents
        
        # 1. Total pool needed to guarantee every survivor gets their step cost (3.0)
        # Note: If wealths are already high, agents don't 'need' the full 3.0.
        # This logic ensures the pool is large enough to save the poorest.
        total_needed = num_alive * self.survival_cost
        total_harvest = sum(raw_rewards[i] for i in alive_indices)
        
        # 2. Determine necessary tax rate (between 0% and 100%)
        # If total harvest is less than needed, we must tax 100% of all earnings.
        required_tax_rate = min(1.0, total_needed / max(0.001, total_harvest))
        
        # 3. Collect from everyone at the required tax rate
        # Zero-sum: Everyone loses their slice, everyone gets an equal dividend
        adjustments = [0.0] * self.n_agents
        
        for i in alive_indices:
            taxed_amount = raw_rewards[i] * required_tax_rate
            adjustments[i] -= taxed_amount
        
        # 4. Redistribute the collected taxes equally
        dividend = (total_harvest * required_tax_rate) / num_alive
        for i in alive_indices:
            adjustments[i] += dividend
            
        return adjustments
    
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
    A Macro-Economic Reinforcement Learning Governor.
    
    Instead of assigning micro-taxes directly, it treats 4 rule-based governors
    as distinct macro-actions. It uses a State-Discretized Policy Gradient 
    (REINFORCE) to learn which economic strategy works best during different 
    phases of the simulation timeline.
    """
    def __init__(self, n_agents=30, max_steps=1000, learning_rate=0.05, death_penalty=100.0, seed=None):
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.death_penalty = death_penalty
        
        # 1. Instantiate the available sub-governors as our discrete action space
        from multi_agent_bandits.core.governor import (
            CommunistGovernor, SocialistGovernor, PigouvianGovernor, FreeMarketGovernor
        )
        self.strategies = [
            CommunistGovernor(n_agents=n_agents),
            SocialistGovernor(n_agents=n_agents, tax_rate=0.5),
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            FreeMarketGovernor(n_agents=n_agents)
        ]
        self.n_actions = len(self.strategies)
        
        # 2. Context Space Setup (Game Phases)
        # Row 0: Early Game, Row 1: Mid Game, Row 2: Late Game
        self.n_states = 3
        self.internal_step = 0  # Clock tracker for temporal categorization
        
        # 3. Policy Parameters: A 2D Matrix of raw preferences (State x Action)
        self.logits_matrix = [[0.0] * self.n_actions for _ in range(self.n_states)]
        
        # 4. Environment framework integration hooks and short-term buffers
        self.last_action = None
        self.last_action_idx = None
        self.last_action_probs = None
        self.last_state_idx = None

        if seed is not None:
            random.seed(seed)

    def _get_state_index(self, observation):
        """
        Determines the current game phase based on timeline progression.
        Increments the internal clock upon selection.
        """
        progress = self.internal_step / self.max_steps
        self.internal_step += 1
        
        if progress < 0.25:
            return 0  # Early Game (0% - 25%)
        elif progress < 0.75:
            return 1  # Mid Game (25% - 75%)
        else:
            return 2  # Late Game (75% - 100%)

    def _get_action_probabilities(self, state_idx):
        """Calculates Softmax percentages specifically for the active game phase row."""
        state_logits = self.logits_matrix[state_idx]
        max_logit = max(state_logits)
        exp_logits = [math.exp(l - max_logit) for l in state_logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """
        Identifies the current game context, samples an economic strategy,
        and delegates tax calculations to that strategy.
        """
        # 1. Identify context phase
        state_idx = self._get_state_index(observation)
        self.last_state_idx = state_idx
        
        # 2. Compute probabilities for this context phase
        probs = self._get_action_probabilities(state_idx)
        self.last_action_probs = probs
        
        # 3. Sample an action index based on weights
        r = random.random()
        cumulative_prob = 0.0
        chosen_idx = 0
        for idx, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                chosen_idx = idx
                break
        self.last_action_idx = chosen_idx
        
        # 4. Delegate computation to the selected structural governor
        selected_strategy = self.strategies[chosen_idx]
        adjustments = selected_strategy.choose_action(
            observation, choices, wealths, raw_rewards, alive_mask
        )
        
        # Framework interface pass-through bypass
        self.last_action = adjustments
        return adjustments

    def update(self, observation, action, reward, next_observation, done=False):
        """
        Performs contextual credit assignment. Nudges the weights of the specific
        matrix cell that generated the current performance score.
        """
        if self.last_action_idx is None or self.last_action_probs is None or self.last_state_idx is None:
            return

        # Downscale step rewards to stabilize exponential gradient updates
        scaled_reward = reward / 100.0 
        state_row = self.last_state_idx

        # REINFORCE update step applied exclusively to the current phase row
        for idx in range(self.n_actions):
            if idx == self.last_action_idx:
                grad = 1.0 - self.last_action_probs[idx]
            else:
                grad = -self.last_action_probs[idx]
            
            # logit = logit + alpha * Scaled_Reward * Gradient
            self.logits_matrix[state_row][idx] += self.learning_rate * scaled_reward * grad

        # Flush action buffers for the next step iteration
        self.last_action_idx = None
        self.last_action_probs = None
        self.last_state_idx = None
        self.last_action = None

    def compute_governor_reward(self, final_rewards, death_count=0):
        """
        Evaluates system performance. Combines raw resource extraction 
        with a structural penalty for enabling bankruptcies.
        """
        total_reward = sum(final_rewards)
        return total_reward - (self.death_penalty * death_count)

    def get_phase_probabilities(self, state_idx):
        """
        Helper method providing a clean percentage string dictionary for console 
        logging wrappers.
        """
        probs = self._get_action_probabilities(state_idx)
        return {
            "Communist": f"{probs[0]*100:.1f}%",
            "Socialist": f"{probs[1]*100:.1f}%",
            "Pigouvian": f"{probs[2]*100:.1f}%",
            "FreeMarket": f"{probs[3]*100:.1f}%"
        }


#NeuralPolicyGradientGovernor:


class PolicyNetwork(nn.Module):
    """
    A lightweight Multi-Layer Perceptron (MLP) mapping continuous 
    environmental state features to a Softmax policy distribution.
    """
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Softmax(dim=-1)  # Guarantees probabilities sum to cleanly 1.0
        )
        
    def forward(self, x):
        return self.network(x)


class NeuralPolicyGradientGovernor:
    """
    Continuous Function Approximation Governor (Deep REINFORCE).
    
    Instead of hardcoded matrix rows/phases, this governor uses a Neural Network
    to look at continuous features (Resource Health, Survivor Ratios, and Inequality)
    and dynamically outputs blended macro-economic policies.
    """
    def __init__(self, n_agents=30, max_steps=1000, learning_rate=0.01, death_penalty=100.0, seed=None):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.death_penalty = death_penalty
        self.learning_rate = learning_rate
        
        # 1. Instantiate underlying structural macro-actions
        from multi_agent_bandits.core.governor import (
            CommunistGovernor, SocialistGovernor, PigouvianGovernor, FreeMarketGovernor
        )
        self.strategies = [
            CommunistGovernor(n_agents=n_agents),
            SocialistGovernor(n_agents=n_agents, tax_rate=0.3),
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            FreeMarketGovernor(n_agents=n_agents)
        ]
        self.n_actions = len(self.strategies)
        
        # 2. Continuous State space features: [Avg Arm Health, Survivor Ratio, Gini Coefficient]
        self.input_dim = 3
        
        # 3. Initialize PyTorch Brain & Adam Optimizer
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        self.policy_net = PolicyNetwork(self.input_dim, self.n_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 4. Step buffers for backpropagation tracing
        self.last_state_tensor = None
        self.last_action_idx = None
        self.last_action = None
        
        # Framework compatibility hooks
        self.internal_step = 0 

    def _calculate_gini(self, wealths):
        """Calculates a quick continuous Gini coefficient tracking system inequality."""
        array = np.array(wealths, dtype=np.float32)
        if np.sum(array) == 0:
            return 0.0
        array += 1e-6  # stability epsilon
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def _extract_continuous_state(self, observation, wealths, alive_mask):
        """Transforms raw environment logs into a clean, normalized feature vector."""
        # Extract arm health arrays appended to the end of your observation vectors
        start_idx = self.n_agents * 2
        arm_healths = observation[start_idx:]
        
        avg_health = np.mean(arm_healths) if arm_healths else 1.0
        survivor_ratio = sum(alive_mask) / len(alive_mask) if alive_mask else 1.0
        gini_inequality = self._calculate_gini(wealths)
        
        # Keep everything tightly bounded between 0.0 and 1.0 for stable gradients
        state_vector = np.array([avg_health, survivor_ratio, gini_inequality], dtype=np.float32)
        return torch.FloatTensor(state_vector)

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """Pipes continuous conditions through the network and samples an economic policy."""
        self.internal_step += 1
        
        # 1. Gather continuous state tensor
        state_tensor = self._extract_continuous_state(observation, wealths, alive_mask)
        self.last_state_tensor = state_tensor
        
        # 2. Feed forward through network to get Softmax probabilities
        probs_tensor = self.policy_net(state_tensor)
        probs = probs_tensor.detach().numpy()
        
        # 3. Categorical sampling based on network confidence
        r = random.random()
        cumulative_prob = 0.0
        chosen_idx = 0
        for idx, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                chosen_idx = idx
                break
                
        self.last_action_idx = chosen_idx
        
        # 4. Delegate computation to the selected policy
        selected_strategy = self.strategies[chosen_idx]
        adjustments = selected_strategy.choose_action(
            observation, choices, wealths, raw_rewards, alive_mask
        )
        
        self.last_action = adjustments
        return adjustments

    def update(self, observation, action, reward, next_observation, done=False):
        """Performs PyTorch automatic backpropagation with Entropy Regularization."""
        if self.last_state_tensor is None or self.last_action_idx is None:
            return

        # 1. Scale step rewards down to prevent weight explosion
        scaled_reward = reward / 100.0
        
        # 2. Re-evaluate computational graph
        probs_tensor = self.policy_net(self.last_state_tensor)
        chosen_prob = probs_tensor[self.last_action_idx]
        
        # 3. Calculate Policy Gradient Loss
        pg_loss = -torch.log(chosen_prob + 1e-8) * scaled_reward
        
        # 🌟 4. NEW: Calculate Entropy Penalty to force exploration
        # Math: H(p) = -sum(p * log(p))
        entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + 1e-8))
        
        # Entropy coefficient (0.01 - 0.05 is standard). 
        # Higher numbers mean the network is forced to stay more curious.
        entropy_coef = 0.02 
        
        # Total Loss: minimize policy loss while maximizing entropy (subtracting it)
        total_loss = pg_loss - (entropy_coef * entropy)
        
        # 5. Run optimization sweep
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping prevents massive steps from freezing the network
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Flush buffers
        self.last_state_tensor = None
        self.last_action_idx = None
        self.last_action = None

    def compute_governor_reward(self, final_rewards, arm_healths, wealths, death_count=0):
        """
        An online-optimized step reward function. 
        Forces the reward magnitude to stay small and tightly bounded, 
        giving the entropy penalty a fair fight at every single step.
        """
        # 1. Economic Production (Scaled down significantly to avoid overwhelming the brain)
        gross_utility = sum(final_rewards) / 100.0  # Scales e.g. 1500.0 down to 15.0
        
        # 2. Ecological Penalty (Triggers the moment resources drop below 70% capacity)
        avg_health = np.mean(arm_healths) if arm_healths else 1.0
        ecology_penalty = 0.0
        if avg_health < 0.70:
            # The lower the health drops, the exponentially worse this penalty becomes
            ecology_penalty = math.pow(0.70 - avg_health, 2) * 50.0 
            
        # 3. Inequality Penalty (Based directly on our continuous Gini score)
        gini = self._calculate_gini(wealths)
        inequality_penalty = 0.0
        if gini > 0.40:
            # Punish the governor for creating severe wealth gaps before agents go bankrupt
            inequality_penalty = (gini - 0.40) * 20.0
            
        # 4. Crisis Death Penalty
        bankruptcy_penalty = death_count * 0.0
        
        # Combine everything into a single localized step score
        total_step_reward = gross_utility - ecology_penalty - inequality_penalty - bankruptcy_penalty
        return total_step_reward

    def get_live_probabilities(self, sample_obs, sample_wealths, sample_alive_mask):
        """Helper method allowing batch execution loggers to inspect current brain outputs."""
        state_tensor = self._extract_continuous_state(sample_obs, sample_wealths, sample_alive_mask)
        with torch.no_grad():
            probs = self.policy_net(state_tensor).numpy()
        return {
            "Communist": f"{probs[0]*100:.1f}%",
            "Socialist": f"{probs[1]*100:.1f}%",
            "Pigouvian": f"{probs[2]*100:.1f}%",
            "FreeMarket": f"{probs[3]*100:.1f}%"
        }
    

#MultiObjectiveNeuralGovernor



class PolicyBrain(nn.Module):
    """A lightweight network that outputs action preferences for a specific objective."""
    def __init__(self, input_dim, output_dim):
        super(PolicyBrain, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        raw_logits = self.fc(x)
        # Clamp to avoid runaway explosion before softmax
        clamped_logits = torch.clamp(raw_logits, min=-4.0, max=4.0)
        return clamped_logits


class MultiObjectiveNeuralGovernor:
    def __init__(self, n_agents=30, learning_rate=0.005, entropy_coef=0.05, seed=None):
        self.n_agents = n_agents
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        
        from multi_agent_bandits.core.governor import (
            CommunistGovernor, SocialistGovernor, PigouvianGovernor, FreeMarketGovernor
        )
        self.strategies = [
            CommunistGovernor(n_agents=n_agents),
            SocialistGovernor(n_agents=n_agents, tax_rate=0.5),
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            FreeMarketGovernor(n_agents=n_agents)
        ]
        self.n_actions = len(self.strategies)
        self.input_dim = 3
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 🌟 THE THREE INDEPENDENT OBJECTIVE BRAINS
        self.brain_economy = PolicyBrain(self.input_dim, self.n_actions)
        self.brain_ecology = PolicyBrain(self.input_dim, self.n_actions)
        self.brain_equality = PolicyBrain(self.input_dim, self.n_actions)
        
        # Optimize all three brains simultaneously
        self.optimizer = optim.Adam(
            list(self.brain_economy.parameters()) + 
            list(self.brain_ecology.parameters()) + 
            list(self.brain_equality.parameters()), 
            lr=self.learning_rate
        )
        
        # Single-step tracking variables
        self.last_state_tensor = None
        self.last_action_idx = None

    def _calculate_gini(self, wealths):
        array = np.array(wealths, dtype=np.float32)
        if np.sum(array) == 0:
            return 0.0
        array += 1e-6
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def _extract_continuous_state(self, arm_healths, wealths, alive_mask):
        avg_health = np.mean(arm_healths) if arm_healths else 1.0
        survivor_ratio = sum(alive_mask) / len(alive_mask) if alive_mask else 1.0
        gini_inequality = self._calculate_gini(wealths)
        
        # Smooth scaling for neural input
        state_vector = np.array([avg_health, survivor_ratio, gini_inequality], dtype=np.float32)
        return torch.FloatTensor(state_vector)

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        # We parse the features directly from the source environment arrays
        start_idx = self.n_agents * 2
        # Cleanly extract arm health from the environment observations vector
        arm_healths = observation[start_idx : start_idx + len(self.strategies)] 
        
        state_tensor = self._extract_continuous_state(arm_healths, wealths, alive_mask)
        self.last_state_tensor = state_tensor
        
        with torch.no_grad():
            logits_econ = self.brain_economy(state_tensor)
            logits_eco = self.brain_ecology(state_tensor)
            logits_eq = self.brain_equality(state_tensor)
            
            # Consensus via Logit Addition: The brains vote together to select the action!
            combined_logits = logits_econ + logits_eco + logits_eq
            probs = torch.softmax(combined_logits, dim=-1).numpy()
            
        chosen_idx = np.random.choice(self.n_actions, p=probs)
        self.last_action_idx = chosen_idx
        
        selected_strategy = self.strategies[chosen_idx]
        return selected_strategy.choose_action(observation, choices, wealths, raw_rewards, alive_mask)

    def compute_vector_rewards(self, final_rewards, arm_healths, wealths, death_count=0):
        """
        Calibrated MORL Vector Engine based on true environment limits.
        Max Score: ~99.3 | Avg Score: ~52.0
        Bounds all channels cleanly between -1.0 and +1.0.
        """
        # 1. Economy Track
        gross_utility = sum(final_rewards) # Real range: ~10.0 to ~99.3
        
        # Center the score around your true empirical average (52.0)
        # If gross_utility is 52.0, centered_econ becomes 0.0
        centered_econ = gross_utility - 52.0
        
        # Scale the variance. Since max is ~99.3, a great turn is +40 above average.
        # Dividing by 40.0 clamps a perfect turn right near +1.0.
        scaled_econ = centered_econ / 40.0
        
        # Apply the death penalty directly to this scaled factor
        # If an agent dies, it drags the economic score down by 0.25 intervals
        raw_econ_signal = scaled_econ - (death_count * 0.25)
        
        # Smoothly lock the final output into the strict democratic [-1.0, +1.0] boundary
        r_economy = math.tanh(raw_econ_signal)
        
        
        # 2. Ecology Track (Unchanged, bounds between -1.0 and +1.0)
        avg_health = np.mean(arm_healths) if arm_healths else 1.0
        if avg_health >= 0.75:
            r_ecology = 1.0
        else:
            r_ecology = max(-1.0, 1.0 - ((0.75 - avg_health) * 4.0))
        
        
        # 3. Equality Track (Unchanged, bounds between -1.0 and +1.0)
        gini = self._calculate_gini(wealths)
        if gini <= 0.30:
            r_equality = 1.0
        else:
            r_equality = max(-1.0, 1.0 - ((gini - 0.30) * 5.0))
            
        return r_economy, r_ecology, r_equality

    def update_morl(self, r_economy, r_ecology, r_equality):
        """Updates each objective brain using ONLY its dedicated tracking reward."""
        if self.last_state_tensor is None or self.last_action_idx is None:
            return
            
        # Re-evaluate the logits for this step
        logits_econ = self.brain_economy(self.last_state_tensor)
        logits_eco = self.brain_ecology(self.last_state_tensor)
        logits_eq = self.brain_equality(self.last_state_tensor)
        
        # Target action index log trackers
        prob_econ = torch.softmax(logits_econ, dim=-1)[self.last_action_idx]
        prob_eco = torch.softmax(logits_eco, dim=-1)[self.last_action_idx]
        prob_eq = torch.softmax(logits_eq, dim=-1)[self.last_action_idx]
        
        # Calculate isolated Loss functions for each distinct brain node
        loss_econ = -torch.log(prob_econ + 1e-8) * r_economy
        loss_eco = -torch.log(prob_eco + 1e-8) * r_ecology
        loss_eq = -torch.log(prob_eq + 1e-8) * r_equality
        
        # Calculate localized entropy to enforce exploration safety boundaries
        total_probs = torch.softmax(logits_econ + logits_eco + logits_eq, dim=-1)
        entropy = -torch.sum(total_probs * torch.log(total_probs + 1e-8))
        
        # Merge individual loss functions into one step execution pass
        total_loss = loss_econ + loss_eco + loss_eq - (self.entropy_coef * entropy)
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain_economy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.brain_ecology.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.brain_equality.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Flush references
        self.last_state_tensor = None
        self.last_action_idx = None

    def get_live_probabilities(self, sample_obs, sample_wealths, sample_alive_mask):
        # Fallback tracking if lists are empty during custom evaluations
        start_idx = self.n_agents * 2
        arm_healths = sample_obs[start_idx : start_idx + self.n_actions]
        state_tensor = self._extract_continuous_state(arm_healths, sample_wealths, sample_alive_mask)
        
        with torch.no_grad():
            combined_logits = self.brain_economy(state_tensor) + self.brain_ecology(state_tensor) + self.brain_equality(state_tensor)
            probs = torch.softmax(combined_logits, dim=-1).numpy()
        return {
            "Communist": f"{probs[0]*100:.1f}%",
            "Socialist": f"{probs[1]*100:.1f}%",
            "Pigouvian": f"{probs[2]*100:.1f}%",
            "FreeMarket": f"{probs[3]*100:.1f}%"
        }
    

#PPONeuralGovernor

class ActorCriticNetwork(nn.Module):
    """
    Factorized Actor-Critic: The Critic head is split into N agent-specific heads 
    to enable individual contribution tracking.
    """
    def __init__(self, input_dim, output_dim, n_agents):
        super(ActorCriticNetwork, self).__init__()
        self.n_agents = n_agents
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor_head = nn.Linear(64, output_dim)
        # Factorized Critic: One head per agent
        self.agent_critic_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_agents)])

    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor_head(features)
        # Individual agent values [batch, n_agents]
        agent_values = torch.cat([head(features) for head in self.agent_critic_heads], dim=-1)
        # Global value for PPO Advantage
        total_value = agent_values.sum(dim=-1, keepdim=True)
        return logits, total_value, agent_values


class PPONeuralGovernor:
    def __init__(self, n_agents=30, learning_rate=0.001, clip_epsilon=0.2, ppo_epochs=4, batch_size=32, seed=None):
        self.n_agents = n_agents
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.max_steps = 50  # Start with short 50-step seasons
        self.success_counter = 0
        self.promotion_threshold = 3 # Promote after 3 "good" episodes

        # strategy performance 
        self.strategy_rewards = np.zeros(4) # Track running reward for each strategy
        self.strategy_counts = np.zeros(4)  # Track how often each was used

        # Initialize core macro strategies
        from multi_agent_bandits.core.governor import (
            SurvivalTargetedGovernor, ProgressiveTaxGovernor, PigouvianGovernor, FreeMarketGovernor
        )
        self.strategies = [
            SurvivalTargetedGovernor(n_agents=n_agents, survival_cost=3.0),
            ProgressiveTaxGovernor(n_agents=n_agents, tax_rate=0.4),
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            FreeMarketGovernor(n_agents=n_agents)
        ]
        self.n_actions = len(self.strategies)
        
        # Continuous state inputs: [Congestion, Survivor Ratio, Poorest Runway, Poverty Rate, Exploitation Focus]
        self.input_dim = 14
        self.last_state_vector = torch.zeros(5)
        self.network = ActorCriticNetwork(self.input_dim, self.n_actions, n_agents)
        
        # --- PARAMETER-SPECIFIC LEARNING RATES ---
        # We split the network parameters into distinct learning speed groups:
        self.optimizer = optim.Adam([
            {'params': self.network.backbone.parameters(), 'lr': learning_rate},
            {'params': self.network.actor_head.parameters(), 'lr': learning_rate},
            # Accelerate the factorized heads specifically
            {'params': self.network.agent_critic_heads.parameters(), 'lr': learning_rate * 3.0} 
        ])
        # ------------------------------------------------

        # 📥 PPO TRAJECTORY MEMORY BUFFER
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_values = []
        
        # Temporary holding parameters for the current single step
        self.current_state_tensor = None
        self.current_action_idx = None
        self.current_log_prob = None

        #logging
        self.input_layer_history = []
        self.output_layer_history = []
        self.reward_layer_history = []
        self.ppo_diagnostics_history = []

    def _calculate_gini(self, wealths):
        # Convert to numpy array
        array = np.array(wealths, dtype=np.float32)
        
        # Filter out dead agents (wealth > 0)
        array = array[array > 0.0]
        
        # Handle edge cases where everyone is dead or everyone has 0 wealth
        if len(array) == 0 or np.sum(array) == 0:
            return 0.0
            
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        return (np.sum((2 * index - len(array) - 1) * array)) / (len(array) * np.sum(array))

    def _extract_state(self, observation, wealths, alive_mask):
        n_alive = sum(alive_mask)
        if n_alive == 0:
            return torch.zeros(5) # Everyone is dead
            
        # --------------------------------------------------------
        # INPUT 1: System-Wide Congestion (HHI Index)
        # --------------------------------------------------------
        # 'observation' contains the agent choices from the previous step
        # Count how many agents picked each arm index
        self.n_arms = len(observation) - (2 * self.n_agents)
        arm_counts = np.zeros(self.n_arms)
        for idx, alive in enumerate(alive_mask):
            if alive:
                # Assuming observation structure allows pulling previous arm chosen
                chosen_arm = observation[idx] 
                arm_counts[chosen_arm] += 1
                
        # Sum of squared proportions
        hhi_congestion = np.sum((arm_counts / n_alive) ** 2)
        # Normalize so that perfectly even distribution yields ~0.0
        min_hhi = 1.0 / self.n_arms
        normalized_congestion = (hhi_congestion - min_hhi) / (1.0 - min_hhi) if self.n_arms > 1 else 1.0
        
        # --------------------------------------------------------
        # INPUT 2: Survivor Ratio
        # --------------------------------------------------------
        survivor_ratio = n_alive / self.n_agents
        
        # --------------------------------------------------------
        # INPUT 3: Poorest Agent Survival Runway
        # --------------------------------------------------------
        living_wealths = [wealths[i] for i in range(self.n_agents) if alive_mask[i]]
        min_wealth = min(living_wealths) if living_wealths else 0.0
        steps_left = min_wealth / 3.0 # step_cost = 3.0
        # Bound it between 0.0 and 1.0 (capped at a safe 20-step runway horizon)
        poorest_runway = min(1.0, steps_left / 20.0)
        
        # --------------------------------------------------------
        # INPUT 4: Poverty Rate (Systemic Risk Headcount)
        # --------------------------------------------------------
        # Let's define the poverty line as having less than 5 steps of life left (15.0 coins)
        poverty_line = 15.0
        poor_count = sum(1 for w in living_wealths if w < poverty_line)
        poverty_rate = poor_count / n_alive
        
        # --------------------------------------------------------
        # INPUT 5: Action Behavioral Entropy (Inferred Exploration)
        # --------------------------------------------------------
        # Measure the randomness of choices on this step as a proxy for exploration
        probs = arm_counts / n_alive
        # Filter out zeros to avoid log(0) errors
        nonzero_probs = probs[probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        max_entropy = np.log2(self.n_arms)
        # High entropy (random choices) means high exploration. 
        # Let's invert it so 1.0 means high optimization/exploitation, and 0.0 means pure random exploration.
        exploitation_focus = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        # --------------------------------------------------------
        # INPUT 6-10: velocity tracking for all 5 indicators (Current - Previous)
        # --------------------------------------------------------
        current_indicators = torch.FloatTensor([
            normalized_congestion, survivor_ratio, poorest_runway, 
            poverty_rate, exploitation_focus
        ])
        
        # Calculate velocity (Change = Current - Previous)
        velocity = current_indicators - self.last_state_vector
        self.last_state_vector = current_indicators.clone() # Update for next time

        #plotting input layer history
        save_values = current_indicators.tolist()
        # WRITE to the internal register immediately
        if not hasattr(self, "input_layer_history"):
            self.input_layer_history = []
        self.input_layer_history.append(save_values)

        # Concatenate 5 base + 5 velocity = 10 inputs
        return torch.cat([current_indicators, velocity])

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        # 1. Extract the 10-length env features
        env_features = self._extract_state(observation, wealths, alive_mask)
        
        # 2. Add the Meta-Data (Strategy History)
        strategy_features = torch.FloatTensor(self.strategy_rewards) / 50.0
        full_state = torch.cat([env_features, strategy_features])
        
        # 3. Perform the inference
        with torch.no_grad():
            logits, total_value, agent_values = self.network(full_state)
            probs = torch.softmax(logits, dim=-1)
            
            # Sampling
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            
        # 4. Store current step data for the buffer
        self.current_state_tensor = full_state
        self.current_action_idx = action_tensor.item()
        self.current_log_prob = dist.log_prob(action_tensor).item()
        self.current_value = total_value.item()

        # 5. Logging (using full_state, not the undefined state_tensor)
        # Note: probs is already computed above, we can reuse it
        self.output_layer_history.append(probs.cpu().numpy().tolist())

        # 6. Get the adjustments from the chosen strategy
        # Ensure we capture the result of the strategy's own choose_action
        strategy_result = self.strategies[self.current_action_idx].choose_action(
            observation, choices, wealths, raw_rewards, alive_mask
        )

        # 7. Explicitly return the action_idx AND the result
        return self.current_action_idx, strategy_result

    def record_step_data(self, gross_rewards, arm_healths, wealths, death_count):
        if self.current_state_tensor is None: return
        
        # Dynamic Social Contract
        progress = min(self.max_steps / 1000.0, 1.0)
        
        r_economy = np.tanh(((sum(gross_rewards) - 52.0) / 40.0) - (death_count * 0.25))
        survivor_ratio = np.sum(np.array(wealths) > 0.0) / self.n_agents
        r_ecology = (1.0 if np.mean(arm_healths) >= 0.75 else max(-1.0, 1.0 - ((0.75 - np.mean(arm_healths)) * 4.0))) * survivor_ratio
        r_equality = (1.0 if self._calculate_gini(wealths) <= 0.30 else max(-1.0, 1.0 - ((self._calculate_gini(wealths) - 0.30) * 5.0))) * survivor_ratio
        
        # Weighted composite: starts with high equity/ecology focus, ends with higher economic focus
        step_reward = (0.2 + 0.5 * progress) * r_economy + \
                      (0.4 - 0.15 * progress) * r_ecology + \
                      (0.4 - 0.35 * progress) * r_equality - (2.0 * death_count)
        
        #testing: isolate economy reward to see if it can learn with a single signal
        step_reward = r_economy

        #plotting reward layer history
        # 2. ADD THIS: Create the register if it doesn't exist
        if not hasattr(self, "reward_layer_history"):
            self.reward_layer_history = []
            
        # 3. ADD THIS: Store the composite AND the components as a list
        # The plotter expects: [Composite, r_economy, r_ecology, r_equality]
        self.reward_layer_history.append([
            step_reward, 
            r_economy, 
            r_ecology, 
            r_equality
        ])

        self.buffer_states.append(self.current_state_tensor)
        self.buffer_actions.append(self.current_action_idx)
        self.buffer_log_probs.append(self.current_log_prob)
        self.buffer_rewards.append(step_reward)
        self.buffer_values.append(self.current_value)
        
        self.current_state_tensor = None # Clear placeholder

    def update_ppo(self):
        """
        🔥 THE PPO OPTIMIZATION ENGINE
        Processes buffered trajectories, computes advantages, and safely updates networks.
        Tracks internal Actor/Critic diagnostic health properties.
        """
        if len(self.buffer_states) == 0:
            return

        # 1. Convert memory lists into comprehensive PyTorch training tensors
        states = torch.stack(self.buffer_states)
        actions = torch.LongTensor(self.buffer_actions)
        old_log_probs = torch.FloatTensor(self.buffer_log_probs)
        old_values = torch.FloatTensor(self.buffer_values)
        rewards = self.buffer_rewards
        
        # 2. Compute discounted rewards-to-go (Targets for Critic evaluation)
        discounted_returns = []
        discounted_sum = 0
        gamma = 0.99  # Long-term economic discount factor
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            discounted_returns.insert(0, discounted_sum)
            
        returns = torch.FloatTensor(discounted_returns)
        # Normalize returns for training variance control
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Temp diagnostic collectors for tracking metrics across this update session
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        epoch_kl_divergences = []

        # 3. PPO Optimization Epoch Loop
        for _ in range(self.ppo_epochs):
            # Generate random batch permutations for stable gradient updates
            permutation = torch.randperm(states.size(0))
            for i in range(0, states.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                if len(indices) == 0:
                    continue

                b_states = states[indices]
                b_actions = actions[indices]
                b_old_log_probs = old_log_probs[indices]
                b_old_values = old_values[indices]
                b_returns = returns[indices]

                # Evaluate current networks status
                logits, state_values, _ = self.network(b_states)
                state_values = state_values.squeeze(-1)
                
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # Extract new log probabilities and entropy tracking profiles
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Calculate localized target Advantage signals
                advantages = b_returns - state_values.detach()

                # Step C: The Probability Ratio r_t(theta)
                ratios = torch.exp(new_log_probs - b_old_log_probs)

                # Step D: Clipped Objective Mathematical Minimization
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss evaluated via Mean Squared Error against raw target returns
                # Penalize the Critic if it jumps too far from its original baseline guess
                value_pred_clipped = b_old_values + torch.clamp(
                    state_values - b_old_values, 
                    -self.clip_epsilon, 
                    self.clip_epsilon
                )
                
                critic_loss_raw = (state_values - b_returns) ** 2
                critic_loss_clipped = (value_pred_clipped - b_returns) ** 2
                
                # Take the maximum structural loss to be safe against variance drops
                critic_loss = 0.5 * torch.max(critic_loss_raw, critic_loss_clipped).mean()
                # -----------------------------------

                # Calculate Approximate KL Divergence for safety/policy-shock tracking
                with torch.no_grad():
                    # Standard formulation: KL(old || new) ≈ ((ratio - 1) - log(ratio))
                    approx_kl = ((ratios - 1) - torch.log(ratios)).mean().item()

                # Aggregate combined objective loss with full critic scaling since we separated the learning rates
                total_loss = actor_loss + critic_loss - (0.01 * entropy)

                # Execute bounded optimization pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Append metrics for this mini-batch update step
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_kl_divergences.append(approx_kl)

        # 4. Global Trajectory Diagnostic Evaluation (Post-Epochs)
        with torch.no_grad():
            _, final_values, _ = self.network(states)
            final_values = final_values.squeeze(-1).cpu().numpy()
            raw_returns = returns.cpu().numpy()
            
            # Compute Explained Variance: EV = 1 - Var(y - y_hat) / Var(y)
            return_variance = np.var(raw_returns)
            explained_variance = (
                1.0 - (np.var(raw_returns - final_values) / return_variance)
                if return_variance > 0 else 0.0
            )

        # Log compiled structural metrics to class container for visual script execution
        self.ppo_diagnostics_history.append({
            "value_loss": np.mean(epoch_critic_losses),
            "policy_loss": np.mean(epoch_actor_losses),
            "entropy": np.mean(epoch_entropies),
            "kl_divergence": np.mean(epoch_kl_divergences),
            "explained_variance": explained_variance
        })

        # 5. Clear the memory buffer for the next trajectory run
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_log_probs.clear()
        self.buffer_rewards.clear()
        self.buffer_values.clear()

    def get_live_probabilities(self, sample_obs, sample_wealths, sample_alive_mask):
        state_tensor = self._extract_state(sample_obs, sample_wealths, sample_alive_mask)
        with torch.no_grad():
            logits, _, _ = self.network(state_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()
        return {
            "Communist": f"{probs[0]*100:.1f}%",
            "Socialist": f"{probs[1]*100:.1f}%",
            "Pigouvian": f"{probs[2]*100:.1f}%",
            "FreeMarket": f"{probs[3]*100:.1f}%"
        }

    def update_season_curriculum(self, current_mse):
        # Only promote if the Critic is actually confident (MSE is low)
        # AND the agent is performing well
        if current_mse < 0.3:  # This threshold can be tuned based on observed training dynamics
            self.success_counter += 1
            if self.success_counter >= self.promotion_threshold:
                print(f"Critic has converged (MSE: {current_mse:.3f}). Promoting to {self.max_steps} steps.")
                self.promote_to_longer_season()
        else:
            # reset the counter if performance is bad
            self.success_counter = 0

    def time_based_update_season_curriculum(self):
        self.success_counter += 1
        if self.success_counter >= self.promotion_threshold:
            print(f"Time-based promotion triggered after {self.success_counter} episodes.")
            self.promote_to_longer_season()

            

            
        #print(f"Current Critic MSE: {current_mse:.3f} | Success Counter: {self.success_counter}/{self.promotion_threshold}")

    def promote_to_longer_season(self):
        self.max_steps = min(1000, self.max_steps + 50)
        self.success_counter = 0 # Reset counter for the next level
        print(f"Promotion! New season length: {self.max_steps}")

    def update_strategy_performance(self, action_idx, reward):
        # Standard incremental average calculation
        self.strategy_counts[action_idx] += 1
        # Alpha (0.1) creates a moving window; older data slowly fades
        alpha = 0.1 
        self.strategy_rewards[action_idx] += alpha * (reward - self.strategy_rewards[action_idx])


class PPOPolicyNetwork(nn.Module):
    """Tiny actor-critic network for the macro-level PPO governor."""
    def __init__(self, input_dim, n_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(64, n_actions)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value


class PPOGovernor:
    """A clean discrete PPO governor that acts every C=10 environment steps."""
    def __init__(
        self,
        n_agents=30,
        n_arms=30,
        decision_interval=10,
        learning_rate=1e-3,
        clip_epsilon=0.2,
        ppo_epochs=4,
        batch_size=32,
        gamma=0.99,
        entropy_coef=0.01,
        seed=None,
    ):
        self.n_agents = n_agents
        self.n_arms = n_arms
        self.decision_interval = decision_interval
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        from multi_agent_bandits.core.governor import (
            PigouvianGovernor, ProgressiveTaxGovernor, SurvivalTargetedGovernor, FreeMarketGovernor
        )

        self.strategies = [
            PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0),
            ProgressiveTaxGovernor(n_agents=n_agents, tax_rate=0.4),
            SurvivalTargetedGovernor(n_agents=n_agents, survival_cost=3.0),
            FreeMarketGovernor(n_agents=n_agents),
        ]
        self.n_actions = len(self.strategies)
        self.network = PPOPolicyNetwork(self.n_arms + 6, self.n_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.current_action_idx = None
        self.current_strategy = self.strategies[0]
        self.current_phase_steps = 0
        self.current_phase_reward = 0.0
        self.internal_step = 0
        self.is_evaluating = False

        self.pending_state = None
        self.pending_value = 0.0
        self.pending_log_prob = 0.0

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_values = []
        self.buffer_rewards = []

        self.input_layer_history = []
        self.output_layer_history = []
        self.reward_layer_history = []

    def reset(self):
        """Reset internal state before a fresh episode."""
        self.current_action_idx = None
        self.current_strategy = self.strategies[0]
        self.current_phase_steps = 0
        self.current_phase_reward = 0.0
        self.internal_step = 0
        self.pending_state = None
        self.pending_value = 0.0
        self.pending_log_prob = 0.0

    def _normalize(self, value, cap):
        return max(0.0, min(1.0, value / cap))

    def _build_global_observation(self, choices, wealths, alive_mask):
        """Convert raw environment signals into a fixed-size global feature vector."""
        n_alive = sum(1 for alive in alive_mask if alive)
        alive_ratio = n_alive / max(1, self.n_agents)

        alive_wealths = [w for w, alive in zip(wealths, alive_mask) if alive]
        mean_wealth = np.mean(alive_wealths) if alive_wealths else 0.0
        wealth_variance = np.var(alive_wealths) if alive_wealths else 0.0
        min_wealth = min(alive_wealths) if alive_wealths else 0.0

        mean_wealth_norm = self._normalize(mean_wealth, 200.0)
        wealth_variance_norm = self._normalize(wealth_variance, 10000.0)
        min_wealth_norm = self._normalize(min_wealth, 200.0)

        arm_counts = np.zeros(self.n_arms, dtype=np.float32)
        for idx, choice in enumerate(choices):
            if alive_mask[idx] and choice is not None and 0 <= choice < self.n_arms:
                arm_counts[choice] += 1.0

        normalized_congestion = arm_counts / max(1.0, n_alive)
        max_congestion = float(np.max(normalized_congestion)) if self.n_arms > 0 else 0.0

        nonzero = normalized_congestion[normalized_congestion > 0.0]
        if len(nonzero) > 0:
            entropy = -np.sum(nonzero * np.log2(nonzero))
            max_entropy = np.log2(self.n_arms) if self.n_arms > 1 else 1.0
            action_diversity = self._normalize(entropy, max_entropy)
        else:
            action_diversity = 0.0

        feature_vector = np.concatenate([
            [alive_ratio, mean_wealth_norm, wealth_variance_norm, min_wealth_norm],
            normalized_congestion,
            [max_congestion, action_diversity]
        ]).astype(np.float32)

        return torch.FloatTensor(feature_vector)

    def choose_action(self, observation, choices, wealths, raw_rewards, alive_mask):
        """Choose a macro policy every decision interval and reuse it for the interval."""
        self.internal_step += 1
        decision_step = ((self.internal_step - 1) % self.decision_interval) == 0

        if self.current_action_idx is None or decision_step:
            state = self._build_global_observation(choices, wealths, alive_mask)
            logits, value = self.network(state)
            probs = torch.softmax(logits, dim=-1)

            if self.is_evaluating:
                action_idx = int(torch.argmax(probs).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action_idx = int(dist.sample().item())

            self.current_action_idx = action_idx
            self.current_strategy = self.strategies[action_idx]
            self.pending_state = state
            self.pending_value = float(value.item())
            self.pending_log_prob = float(torch.log(probs[action_idx] + 1e-8).item())

            self.input_layer_history.append(state.tolist())
            self.output_layer_history.append(probs.detach().cpu().numpy().tolist())

        adjustments = self.current_strategy.choose_action(
            observation, choices, wealths, raw_rewards, alive_mask
        )

        return self.current_action_idx, adjustments

    def record_step_data(self, gross_rewards, arm_healths, wealths, death_count):
        """Accumulate step rewards and create PPO transitions every decision interval."""
        self.current_phase_reward += sum(gross_rewards)
        self.current_phase_steps += 1

        if self.current_phase_steps >= self.decision_interval and self.pending_state is not None:
            self.buffer_states.append(self.pending_state)
            self.buffer_actions.append(self.current_action_idx)
            self.buffer_log_probs.append(self.pending_log_prob)
            self.buffer_values.append(self.pending_value)
            self.buffer_rewards.append(self.current_phase_reward)
            self.reward_layer_history.append([
                self.current_phase_reward,
                self.current_phase_reward,
                0.0,
                0.0,
            ])
            self.current_phase_steps = 0
            self.current_phase_reward = 0.0

    def _flush_partial_interval(self):
        if self.current_phase_steps > 0 and self.pending_state is not None:
            self.buffer_states.append(self.pending_state)
            self.buffer_actions.append(self.current_action_idx)
            self.buffer_log_probs.append(self.pending_log_prob)
            self.buffer_values.append(self.pending_value)
            self.buffer_rewards.append(self.current_phase_reward)
            self.reward_layer_history.append([
                self.current_phase_reward,
                self.current_phase_reward,
                0.0,
                0.0,
            ])
            self.current_phase_steps = 0
            self.current_phase_reward = 0.0

    def update_ppo(self):
        """Run the PPO optimization pass over collected interval-level transitions."""
        self._flush_partial_interval()

        if len(self.buffer_states) == 0:
            return

        states = torch.stack(self.buffer_states)
        actions = torch.tensor(self.buffer_actions, dtype=torch.int64)
        old_log_probs = torch.tensor(self.buffer_log_probs, dtype=torch.float32)
        old_values = torch.tensor(self.buffer_values, dtype=torch.float32)
        rewards = self.buffer_rewards

        returns = []
        discounted_sum = 0.0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(states.size(0))
            for start in range(0, states.size(0), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                logits, values = self.network(batch_states)
                values = values.squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                surrogate1 = ratios * batch_advantages
                surrogate2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = actor_loss + critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()
        self.buffer_rewards.clear()

    def get_live_probabilities(self, sample_obs, sample_wealths, sample_alive_mask):
        """Return a human-readable probability dictionary for debugging."""
        choices = [int(x) if x is not None and x >= 0 else None for x in sample_obs[:self.n_agents]]
        wealths = list(sample_wealths) if sample_wealths is not None else list(sample_obs[self.n_agents:2*self.n_agents])
        state = self._build_global_observation(choices, wealths, sample_alive_mask)
        with torch.no_grad():
            logits, _ = self.network(state)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {
            "Pigouvian": f"{probs[0]*100:.1f}%",
            "Progressive": f"{probs[1]*100:.1f}%",
            "Survival": f"{probs[2]*100:.1f}%",
            "FreeMarket": f"{probs[3]*100:.1f}%",
        }

    def compute_governor_reward(self, final_rewards, death_count=0):
        return sum(final_rewards)
