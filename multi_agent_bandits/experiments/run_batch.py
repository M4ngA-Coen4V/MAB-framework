import numpy as np
import os
import sys
import torch
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import (
    PigouvianGovernor, CommunistGovernor, SocialistGovernor, LearningGovernorAI, 
    SafetyNetGovernor, FreeMarketGovernor, DynamicTaxingGovernor, 
    NeuralPolicyGradientGovernor, MultiObjectiveNeuralGovernor, PPONeuralGovernor
)
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.core.experiment_runner import ExperimentRunner

def run_phase(governor, n_trials, steps, n_agents, is_training=True):
    """Helper function to execute a batch phase cleanly."""
    batch_total_rewards = []
    batch_survival_counts = []
    batch_extinction_steps = []
    
    original_stdout = sys.stdout
    phase_name = "Training" if is_training else "Testing"

    for trial in range(n_trials):
        env = DepletingCommonsEnvironment(
            n_agents=n_agents,
            death_threshold=0.0,
            initial_wealth=200.0,
            step_cost=3.0,
            governor=governor
        )
        
        agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)
        
        try:
            sys.stdout = open(os.devnull, 'w')  # Mute stdout
            runner.run(
                plot_rewards=False, plot_frequencies=False, plot_beliefs=False, 
                plot_environment_health=False, plot_resource_efficiency=False
            )
        finally:
            sys.stdout = original_stdout  # Unmute

        # Training Check: Only update weights if the governor supports it and we are training
        if is_training and governor is not None and hasattr(governor, "update_ppo"):
            governor.update_ppo()
        elif governor is not None and hasattr(governor, "buffer_states"):
            governor.buffer_states.clear()
            governor.buffer_actions.clear()
            governor.buffer_log_probs.clear()
            governor.buffer_rewards.clear()

        # Gather Statistics
        batch_total_rewards.append(sum(runner.total_rewards))
        survivor_count = sum(1 for step in env.death_steps if step is None)
        batch_survival_counts.append(survivor_count)
        
        if survivor_count == 0:
            extinction_step = max([step for step in env.death_steps if step is not None])
            batch_extinction_steps.append(extinction_step)

        # 📊 LIVE PROBABILITY DIAGNOSTIC PRINT
        # Prints diagnostic states for both Training and Testing phases
        if (trial + 1) % 10 == 0:
            print(f" -> [{phase_name}] Completed {trial + 1}/{n_trials} trials...")
            if governor is not None and hasattr(governor, "get_live_probabilities") and is_training:
                mock_obs = [0] * (governor.n_agents * 2) + [1.0] * env.n_arms
                mock_wealths = [50.0] * governor.n_agents
                mock_alive = [True] * governor.n_agents
                
                # 1. Pristine State
                pristine_probs = governor.get_live_probabilities(mock_obs, mock_wealths, mock_alive)
                # 2. Ecologically Stressed State
                stressed_obs = [0] * (governor.n_agents * 2) + [0.3] * env.n_arms
                stressed_probs = governor.get_live_probabilities(stressed_obs, mock_wealths, mock_alive)
                # 3. Inequality Crisis State
                unequal_wealths = [500.0 if idx < 5 else 2.0 for idx in range(governor.n_agents)]
                unequal_probs = governor.get_live_probabilities(mock_obs, unequal_wealths, mock_alive)
                
                print(f"     If Commons Pristine   -> Communist: {pristine_probs['Communist']}, Socialist: {pristine_probs['Socialist']}, Pigouvian: {pristine_probs['Pigouvian']}, FreeMarket: {pristine_probs['FreeMarket']}")
                print(f"     If Resource Depleted  -> Communist: {stressed_probs['Communist']}, Socialist: {stressed_probs['Socialist']}, Pigouvian: {stressed_probs['Pigouvian']}, FreeMarket: {stressed_probs['FreeMarket']}")
                print(f"     If Inequality Spikes  -> Communist: {unequal_probs['Communist']}, Socialist: {unequal_probs['Socialist']}, Pigouvian: {unequal_probs['Pigouvian']}, FreeMarket: {unequal_probs['FreeMarket']}\n")

    return {
        "avg_reward": np.mean(batch_total_rewards),
        "std_reward": np.std(batch_total_rewards),
        "max_reward": np.max(batch_total_rewards),
        "min_reward": np.min(batch_total_rewards),
        "avg_survivors": np.mean(batch_survival_counts),
        "extinction_rate": (len(batch_extinction_steps) / n_trials) * 100,
        "extinction_steps": batch_extinction_steps
    }

def run_batch_simulation(n_trials=100, steps=1000):
    n_agents = 30
    
    governor = PPONeuralGovernor(n_agents=n_agents, learning_rate=0.001, clip_epsilon=0.2, ppo_epochs=4, seed=None)   
    governor_name = governor.__class__.__name__ if governor is not None else "None (Baseline)"
    is_learning_model = hasattr(governor, "update_ppo")

    # Save the original choose_action to keep things clean
    original_choose_action = governor.choose_action

    # ========================================================
    # PHASE 1: TRAINING BLOCK
    # ========================================================
    print(f"🚀 PHASE 1: Training {governor_name} over {n_trials} trials...\n")
    train_metrics = run_phase(governor, n_trials, steps, n_agents, is_training=True)
    print("✅ Training complete. Network weights locked.\n")

    # ========================================================
    # PHASE 2: TESTING / EVALUATION BLOCK (100 Trials)
    # ========================================================
    print(f"🧪 PHASE 2: Evaluating final {governor_name} performance across {n_trials} independent test games...\n")
    
    def native_sampling_choose_action(observation, choices, wealths, raw_rewards, alive_mask):
        state_tensor = governor._extract_state(observation, wealths, alive_mask)
        with torch.no_grad(): # Keeps weights locked
            logits, _ = governor.network(state_tensor)
            probs = torch.softmax(logits, dim=-1) # Use raw learned probabilities
            
            dist = torch.distributions.Categorical(probs)
            chosen_idx = dist.sample().item() # Sample naturally from the distribution!
            
        return governor.strategies[chosen_idx].choose_action(observation, choices, wealths, raw_rewards, alive_mask)
    
    # Swap behavioral hook to use natural sampling
    governor.choose_action = native_sampling_choose_action
    
    test_metrics = run_phase(governor, n_trials, steps, n_agents, is_training=False)

    # Restore original method footprint just in case
    governor.choose_action = original_choose_action

    # ========================================================
    # COMPREHENSIVE SIMULATION REPORT
    # ========================================================
    print("\n" + "="*60)
    print(f"📊 SYSTEM SUMMARY REPORT: {governor_name}")
    print("="*60)
    print(f"| Metric                      | Training Phase      | Testing Phase       |")
    print(f"|-----------------------------|---------------------|---------------------|")
    print(f"| Avg Combined System Reward  | {train_metrics['avg_reward']:.2f} ± {train_metrics['std_reward']:.2f}  | {test_metrics['avg_reward']:.2f} ± {test_metrics['std_reward']:.2f}  |")
    print(f"| Max Combined System Reward  | {train_metrics['max_reward']:.2f}            | {test_metrics['max_reward']:.2f}            |")
    print(f"| Min Combined System Reward  | {train_metrics['min_reward']:.2f}            | {test_metrics['min_reward']:.2f}            |")
    print(f"| Avg Active Survivors / 30   | {train_metrics['avg_survivors']:.1f}                  | {test_metrics['avg_survivors']:.1f}                  |")
    print(f"| Complete Extinction Rate    | {train_metrics['extinction_rate']:.1f}%                | {test_metrics['extinction_rate']:.1f}%                |")
    
    if train_metrics['extinction_steps'] or test_metrics['extinction_steps']:
        t_ext = f"{np.mean(train_metrics['extinction_steps']):.1f}" if train_metrics['extinction_steps'] else "N/A"
        e_ext = f"{np.mean(test_metrics['extinction_steps']):.1f}" if test_metrics['extinction_steps'] else "N/A"
        print(f"| Avg Extinction Timestep     | {t_ext:<19} | {e_ext:<19} |")
    print("="*60)

    # ========================================================
    # 🌟 PHASE 3: VISUAL DIAGNOSTIC RUN (Saves to test_PPO)
    # ========================================================
    print("\n📸 PHASE 3: Launching single visual tracking trial...")
    print("Saving all system charts to destination directory: './test_PPO' ...")

    # Re-initialize a fresh environment hooked up to our trained greedy governor
    visual_env = DepletingCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=0.0,
        initial_wealth=200.0,
        step_cost=3.0,
        governor=governor
    )
    visual_agents = [EpsilonGreedyAgent(visual_env.n_arms) for _ in range(n_agents)]
    
    # Run cleanly with standard output active, saving graphs to target folder
    visual_runner = ExperimentRunner(
        visual_env, 
        visual_agents, 
        timestep_limit=steps, 
        save_dir="test_PPO"
    )
    
    visual_runner.run(
        plot_rewards=True, 
        plot_frequencies=True, 
        plot_beliefs=True, 
        plot_environment_health=True, 
        plot_resource_efficiency=True
    )

    # Flush the evaluation buffer out to ensure no memory residue leaks
    if hasattr(governor, "buffer_states"):
        governor.buffer_states.clear()
        governor.buffer_actions.clear()
        governor.buffer_log_probs.clear()
        governor.buffer_rewards.clear()

    # Restore native choose_action pattern out of good practice
    governor.choose_action = original_choose_action
    print("🎉 Visual diagnostic complete! Check your new './test_PPO' directory for the PNG outputs.")

if __name__ == "__main__":
    run_batch_simulation(n_trials=100, steps=1000)