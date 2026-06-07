import numpy as np
import os
import sys
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import PigouvianGovernor, CommunistGovernor, SocialistGovernor, LearningGovernorAI, SafetyNetGovernor, FreeMarketGovernor, DynamicTaxingGovernor
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
    governor = DynamicTaxingGovernor(n_agents=n_agents, learning_rate=0.05, death_penalty=0.0)

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
        if (trial + 1) % 10 == 0:
            print(f"  -> Completed {trial + 1}/{n_trials} trials...")
            
            # 1. Pull the percentage dictionaries for each phase
            early_probs = governor.get_phase_probabilities(0)
            mid_probs   = governor.get_phase_probabilities(1)
            late_probs  = governor.get_phase_probabilities(2)
            
            # 2. Print them as beautifully aligned rows
            print(f"     Early-Game (Wealth Build) -> Communist: {early_probs['Communist']}, Socialist: {early_probs['Socialist']}, Pigouvian: {early_probs['Pigouvian']}, FreeMarket: {early_probs['FreeMarket']}")
            print(f"     Mid-Game   (Stabilization) -> Communist: {mid_probs['Communist']}, Socialist: {mid_probs['Socialist']}, Pigouvian: {mid_probs['Pigouvian']}, FreeMarket: {mid_probs['FreeMarket']}")
            print(f"     Late-Game  (Crisis/Survival)-> Communist: {late_probs['Communist']}, Socialist: {late_probs['Socialist']}, Pigouvian: {late_probs['Pigouvian']}, FreeMarket: {late_probs['FreeMarket']}\n")

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