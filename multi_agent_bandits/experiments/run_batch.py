import numpy as np
import os
import sys
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import PigouvianGovernor, CommunistGovernor, SocialistGovernor, LearningGovernorAI, SafetyNetGovernor, FreeMarketGovernor, DynamicTaxingGovernor, NeuralPolicyGradientGovernor, MultiObjectiveNeuralGovernor
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.core.experiment_runner import ExperimentRunner

def run_batch_simulation(n_trials=100, steps=1000):
    n_agents = 30
    
    # Storage arrays for statistical aggregation
    batch_total_rewards = []
    batch_final_wealths = []
    batch_survival_counts = []
    batch_extinction_steps = []

    print(f"🚀 Starting Batch Execution: Running {n_trials} independent trials...")

    # Redirect standard output momentarily to suppress individual experiment spam
    original_stdout = sys.stdout
    governor_name = "None (Baseline)"

    #governor = DynamicTaxingGovernor(n_agents=n_agents, learning_rate=0.05, death_penalty=0.0)
    #governor = NeuralPolicyGradientGovernor(n_agents=n_agents, max_steps=1000, learning_rate=0.01, death_penalty=0.0)
    governor = MultiObjectiveNeuralGovernor(n_agents=n_agents, learning_rate=0.005, entropy_coef=0.05, seed=None)   

    for trial in range(n_trials):
        # 1. ALWAYS initialize a completely fresh env, governor, and agents per trial
        #governor = PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=10.0)
        #governor = None
        #governor = FreeMarketGovernor()
        #governor = LearningGovernorAI(n_agents=n_agents)
        #governor = CommunistGovernor(n_agents=n_agents)
        #governor = SocialistGovernor(n_agents=n_agents, tax_rate=0.1)
        #governor = SafetyNetGovernor(n_agents=n_agents, safety_buffer=0.0, step_cost=3.0)
        #governor = DynamicTaxingGovernor(n_agents=n_agents, learning_rate=0.05, death_penalty=0.0)

        # Dynamically capture the name of the active governor once
        if governor is not None and trial == 0:
            governor_name = governor.__class__.__name__

        env = DepletingCommonsEnvironment(
            n_agents=n_agents,
            death_threshold=0.0,
            initial_wealth=200.0,
            step_cost=3.0,
            governor=governor
        )
        
        agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]
        
        # 2. Quietly run the experiment runner without generating individual charts
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)
        
        try:
            sys.stdout = open(os.devnull, 'w') # Mute standard outputs
            runner.run(
                plot_rewards=False, 
                plot_frequencies=False, 
                plot_beliefs=False, 
                plot_environment_health=False, 
                plot_resource_efficiency=False
            )
        finally:
            sys.stdout = original_stdout # Unmute immediately after step completion

        # 3. Collect statistics from the finished trial
        trial_combined_reward = sum(runner.total_rewards)
        trial_combined_wealth = sum(env.agent_wealths)
        
        # Count who managed to survive the full run
        survivor_count = sum(1 for step in env.death_steps if step is None)
        
        batch_total_rewards.append(trial_combined_reward)
        batch_final_wealths.append(trial_combined_wealth)
        batch_survival_counts.append(survivor_count)
        
        # Track when extinction occurred (if at all)
        if survivor_count == 0:
            # If everyone died, find the latest step someone was alive
            extinction_step = max([step for step in env.death_steps if step is not None])
            batch_extinction_steps.append(extinction_step)

        # Progress indicator
        # Inside run_batch.py trial loop tracking blocks
        if (trial + 1) % 10 == 0:
            print(f" -> Completed {trial + 1}/{n_trials} trials...")
            
            # Create a mock environment footprint to ask the brain for a prediction readout
            mock_obs = [0] * (governor.n_agents * 2) + [1.0] * 30 # 30 arms at health 1.0
            mock_wealths = [50.0] * governor.n_agents
            mock_alive = [True] * governor.n_agents
            
            # 1. Query Pristine State (High health, complete parity)
            pristine_probs = governor.get_live_probabilities(mock_obs, mock_wealths, mock_alive)
            
            # 2. Query Ecologically Stressed State (Low arm health)
            stressed_obs = [0] * (governor.n_agents * 2) + [0.3] * 30 # Arms dropped to 30% capacity
            stressed_probs = governor.get_live_probabilities(stressed_obs, mock_wealths, mock_alive)
            
            # 3. Query Wealth Inequality Inequality Crisis State (Wiped out bottom earners)
            unequal_wealths = [500.0 if idx < 5 else 2.0 for idx in range(governor.n_agents)]
            unequal_probs = governor.get_live_probabilities(mock_obs, unequal_wealths, mock_alive)
            
            print(f"     If Commons Pristine   -> Communist: {pristine_probs['Communist']}, Socialist: {pristine_probs['Socialist']}, Pigouvian: {pristine_probs['Pigouvian']}, FreeMarket: {pristine_probs['FreeMarket']}")
            print(f"     If Resource Depleted  -> Communist: {stressed_probs['Communist']}, Socialist: {stressed_probs['Socialist']}, Pigouvian: {stressed_probs['Pigouvian']}, FreeMarket: {stressed_probs['FreeMarket']}")
            print(f"     If Inequality Spikes  -> Communist: {unequal_probs['Communist']}, Socialist: {unequal_probs['Socialist']}, Pigouvian: {unequal_probs['Pigouvian']}, FreeMarket: {unequal_probs['FreeMarket']}\n")

    # --- Step 4: Compute Statistical Metrics ---
    avg_reward = np.mean(batch_total_rewards)
    std_reward = np.std(batch_total_rewards)
    max_reward = np.max(batch_total_rewards)
    min_reward = np.min(batch_total_rewards)
    
    avg_wealth = np.mean(batch_final_wealths)
    avg_survivors = np.mean(batch_survival_counts)
    
    extinction_rate = (len(batch_extinction_steps) / n_trials) * 100

    # Output the structured final digest complete with new requested parameters
    print("\n" + "="*50)
    print(f"📊 BATCH SIMULATION SUMMARY ({n_trials} Trials)")
    print("="*50)
    print(f"| Metric                      | Value                  |")
    print(f"|-----------------------------|------------------------|")
    print(f"| Active Policy Governor      | {governor_name:<22} |")
    print(f"| Avg Combined System Reward  | {avg_reward:.2f} ± {std_reward:.2f} |")
    print(f"| Max Combined System Reward  | {max_reward:.2f}            |")
    print(f"| Min Combined System Reward  | {min_reward:.2f}            |")
    #print(f"| Avg Combined Final Wealth   | {avg_wealth:.2f}            |")
    print(f"| Avg Active Survivors / 30   | {avg_survivors:.1f}             |")
    print(f"| Complete Extinction Rate    | {extinction_rate:.1f}%            |")
    if batch_extinction_steps:
        print(f"| Avg Extinction Timestep     | {np.mean(batch_extinction_steps):.1f}            |")
    print("="*50)

if __name__ == "__main__":
    run_batch_simulation(n_trials=100, steps=1000)