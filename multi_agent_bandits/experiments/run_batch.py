import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import (
    PigouvianGovernor, CommunistGovernor, SocialistGovernor, LearningGovernorAI, 
    SafetyNetGovernor, FreeMarketGovernor, DynamicTaxingGovernor, 
    NeuralPolicyGradientGovernor, MultiObjectiveNeuralGovernor, PPONeuralGovernor,
    ProgressiveTaxGovernor, SurvivalTargetedGovernor
)
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.core.experiment_runner import ExperimentRunner

#plotting fucntions
def plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends.png"):
    """Generates a comprehensive line graph tracking the 5 dashboard inputs over time."""
    if not hasattr(governor, "input_layer_history") or len(governor.input_layer_history) == 0:
        print("⚠️ No input layer history found to plot.")
        return

    # Convert the list of rows into an array for easy slicing: shape (Timesteps, 5)
    history_matrix = np.array(governor.input_layer_history)
    timesteps = range(len(history_matrix))

    labels = [
        "Normalized Congestion",
        "Survivor Ratio",
        "Poorest Runway",
        "Poverty Rate",
        "Exploitation Focus"
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    plt.figure(figsize=(12, 6))
    
    # Plot each of the 5 input layer lines
    for i in range(5):
        plt.plot(timesteps, history_matrix[:, i], label=labels[i], color=colors[i], alpha=0.85, linewidth=2)

    plt.title(f"PPO Governor Input Layer Dashboard Dynamics Over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Simulation Timestep", fontsize=12)
    plt.ylabel("Normalized Indicator Metric Value (0.0 to 1.0)", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Dashboard input tracking plot successfully saved to: '{save_path}'")

def plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends.png"):
    """Generates an overlapping line chart tracking the exact probability weights of the 4 macro governors."""
    if not hasattr(governor, "output_layer_history") or len(governor.output_layer_history) == 0:
        print("⚠️ No output layer history found to plot.")
        return

    # Convert tracker list into a structured NumPy array: shape (Timesteps, 4)
    history_matrix = np.array(governor.output_layer_history)
    timesteps = range(len(history_matrix))

    labels = ["SurvivalTargeted", "Progressive", "Pigouvian", "FreeMarket"]
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"] # Red, Green, Orange, Blue

    plt.figure(figsize=(12, 6))
    
    # Plot individual overlapping lines
    for i in range(4):
        plt.plot(
            timesteps, 
            history_matrix[:, i], 
            label=labels[i], 
            color=colors[i], 
            linewidth=2.5, 
            alpha=0.9
        )

    plt.title("PPO Governor Strategy Allocation (Output Layer Ratios) Over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Simulation Timestep", fontsize=12)
    plt.ylabel("Policy Selection Probability Ratio (0.0 to 1.0)", fontsize=12)
    
    plt.xlim(0, len(history_matrix) - 1)
    plt.ylim(-0.02, 1.02) # Slightly padded to ensure lines at 0.0 or 1.0 aren't cut off
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=11)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Strategy line tracking plot successfully saved to: '{save_path}'")


def plot_reward_decomposition_isolated(governor, save_path="test_PPO/reward_decomposition.png"):
    """
    Generates a standalone, publication-quality plot displaying the 
    Composite Total Reward alongside its 3 underlying unweighted sub-rewards.
    """
    if not hasattr(governor, "reward_layer_history") or len(governor.reward_layer_history) == 0:
        print("⚠️ No reward history found for plotting.")
        return

    # Convert the logged history list into a structured numpy array
    reward_matrix = np.array(governor.reward_layer_history)
    timesteps = range(len(reward_matrix))

    # Create a single, clear figure
    plt.figure(figsize=(12, 6.5))

    # Map arrays based on your record_step_data append sequence:
    # index 0 -> step_reward (weighted composite)
    # index 1 -> r_economy (raw unweighted)
    # index 2 -> r_ecology (raw unweighted)
    # index 3 -> r_equality (raw unweighted)
    plt.plot(timesteps, reward_matrix[:, 0], color='#000000', label='Total Step Reward (Composite)', linewidth=3, linestyle='-')
    plt.plot(timesteps, reward_matrix[:, 1], color='#9467bd', label='Economic Performance ($r_{econ}$)', linewidth=2, linestyle='--')
    plt.plot(timesteps, reward_matrix[:, 2], color='#8c564b', label='Ecological Health ($r_{ecol}$)', linewidth=2, linestyle='-.')
    plt.plot(timesteps, reward_matrix[:, 3], color='#e377c2', label='Social Equality ($r_{eq}$)', linewidth=2, linestyle=':')

    # Styling and Axis Configuration
    plt.title("Reward Function Components & Scalar Decomposition", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Simulation Timestep", fontsize=12, fontweight='semibold')
    plt.ylabel("Raw Metric Value (-1.0 to 1.0)", fontsize=12, fontweight='semibold')
    
    plt.xlim(0, len(timesteps))
    plt.ylim(-1.05, 1.05)  # Enforce strict standard scalar bounds
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', frameon=True, shadow=True, facecolor='white', edgecolor='none')

    # Ensure output directories exist and save cleanly
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"📊 Isolated Reward Decomposition Plot successfully saved to: '{save_path}'")

def plot_actor_critic_diagnostics(diagnostics_history, save_dir="./test_PPO"):
    """
    Generates a 2x2 multi-panel diagnostic dashboard tracking 
    the internal optimization health of the Actor and Critic networks.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data series
    trials = np.arange(1, len(diagnostics_history) + 1)
    v_loss = [d["value_loss"] for d in diagnostics_history]
    p_loss = [d["policy_loss"] for d in diagnostics_history]
    entropy = [d["entropy"] for d in diagnostics_history]
    kl_div = [d["kl_divergence"] for d in diagnostics_history]
    exp_var = [d["explained_variance"] for d in diagnostics_history]
    
    # Initialize the academic 2x2 grid style dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Governor Network Internal Health Dashboard", fontsize=16, fontweight='bold', y=0.96)
    
    # --------------------------------------------------------
    # Panel 1: Critic Value Loss & Predicted Variance Accuracy
    # --------------------------------------------------------
    ax1 = axs[0, 0]
    color = 'tab:red'
    ax1.set_xlabel('Training Iteration (Updates)', fontweight='bold')
    ax1.set_ylabel('Critic MSE Value Loss', color=color, fontweight='bold')
    ax1.plot(trials, v_loss, color=color, alpha=0.8, linewidth=2, label="Value Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Instantiate a second y-axis sharing the same x-axis for Explained Variance
    ax1_twin = ax1.twinx()
    color = 'tab:green'
    ax1_twin.set_ylabel('Explained Variance (Target: 1.0)', color=color, fontweight='bold')
    ax1_twin.plot(trials, exp_var, color=color, linestyle=":", alpha=0.9, linewidth=2, label="Explained Var")
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1_twin.axhline(0.0, color='black', linestyle='-', alpha=0.3) # 0.0 threshold anchor
    ax1.set_title("Critic Performance Evaluation", fontsize=12, fontweight='bold')

    # --------------------------------------------------------
    # Panel 2: Policy Exploration Entropy (The System Urgency Gauge)
    # --------------------------------------------------------
    ax2 = axs[0, 1]
    ax2.plot(trials, entropy, color='purple', linewidth=2, label="Action Entropy")
    ax2.set_xlabel('Training Iteration (Updates)', fontweight='bold')
    ax2.set_ylabel('Shannon Entropy H(π)', fontweight='bold')
    ax2.set_title("Actor Exploration / Policy Randomness", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # --------------------------------------------------------
    # Panel 3: Step Policy Update Distance (KL Divergence Safe-Zones)
    # --------------------------------------------------------
    ax3 = axs[1, 0]
    ax3.plot(trials, kl_div, color='darkorange', linewidth=2, label="Approx. KL")
    ax3.set_xlabel('Training Iteration (Updates)', fontweight='bold')
    ax3.set_ylabel('Approximate KL Divergence', fontweight='bold')
    ax3.set_title("Policy Update Shift Distance (KL)", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    # Highlight a standard target safety roof line (typically around 0.015 - 0.03)
    ax3.axhline(0.02, color='red', linestyle='--', alpha=0.7, label='Target Constraint Ceiling')
    ax3.legend(loc='upper right')

    # --------------------------------------------------------
    # Panel 4: Policy Optimization Step Gain Trend
    # --------------------------------------------------------
    ax4 = axs[1, 1]
    ax4.plot(trials, p_loss, color='dodgerblue', linewidth=2, label="Policy Loss")
    ax4.set_xlabel('Training Iteration (Updates)', fontweight='bold')
    ax4.set_ylabel('Surrogate Objective Loss', fontweight='bold')
    ax4.set_title("Actor Policy Advantage Gradient Direction", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.axhline(0.0, color='black', linestyle='-', alpha=0.3)

    # Clean layout wrapping up
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(save_dir, "actor_critic_health.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Internal Actor-Critic structural dashboard saved to: '{save_path}'")


#run batch
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

def run_batch_simulation(train_n_trials=100, test_n_trials=100, steps=1000):
    n_agents = 30
    
    #governor = ProgressiveTaxGovernor(n_agents=n_agents, tax_rate=0.4)
    #governor = PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=20.0)
    #governor = SurvivalTargetedGovernor(n_agents=n_agents, survival_cost=3.0)

    governor = PPONeuralGovernor(n_agents=n_agents, learning_rate=0.001, clip_epsilon=0.2, ppo_epochs=4, batch_size=1000, seed=None)
    #    
    governor_name = governor.__class__.__name__ if governor is not None else "None (Baseline)"
    is_learning_model = hasattr(governor, "update_ppo")

    # --- ADD THIS DEFAULT INITIALIZATION ---
    train_metrics = {
        "avg_reward": 0.0, "std_reward": 0.0, "max_reward": 0.0, 
        "min_reward": 0.0, "avg_survivors": 0.0, "extinction_rate": 0.0,
        "extinction_steps": []
    }
    # ---------------------------------------

    # Save the original choose_action to keep things clean
    original_choose_action = governor.choose_action

    # ========================================================
    # PHASE 1: TRAINING BLOCK
    # ========================================================
    if is_learning_model:
        print(f"🚀 PHASE 1: Training {governor_name} over {train_n_trials} trials...\n")
        # We loop through trials, but the 'max_steps' inside changes over time!
        for trial in range(train_n_trials):
            # Run a single trial of the current "Season Length"
            train_metrics = run_phase(governor, n_trials=1, steps=governor.max_steps, n_agents=n_agents, is_training=True)
            
            # After each trial, check if the Governor earned a promotion
            # (Assuming you added the check_episode_success method to governor)
            # We use a placeholder for alive_ratio and gini from the trial results
            # You'll need to pass those back from run_phase or track them here
            true_mse = governor.ppo_diagnostics_history[-1]["value_loss"] * 2
            #governor.update_season_curriculum(true_mse)
            governor.time_based_update_season_curriculum()

            if (trial + 1) % 10 == 0:
                print(f"Trial {trial+1}/{train_n_trials} | Current Season Length: {governor.max_steps}")

        print("✅ Training complete. Network weights locked.\n")

        # ---------------------------------------------------------------------
        # 📊 NEW: PLOT ACTOR-CRITIC INTERNAL HEALTH
        # ---------------------------------------------------------------------
        print("📊 Generating Actor-Critic structural health diagnostic dashboard...")

        # Pass the accumulated diagnostics array from inside your governor instance
        plot_actor_critic_diagnostics(diagnostics_history=governor.ppo_diagnostics_history, save_dir="./test_PPO")

    # ========================================================
    # PHASE 2: TESTING / EVALUATION BLOCK (100 Trials)
    # ========================================================
    print(f"🧪 PHASE 2: Evaluating final {governor_name} performance across {test_n_trials} independent test games...\n")
    
    def native_sampling_choose_action(observation, choices, wealths, raw_rewards, alive_mask):
        state_tensor = governor._extract_state(observation, wealths, alive_mask)
        with torch.no_grad(): # Keeps weights locked
            logits, _ = governor.network(state_tensor)
            probs = torch.softmax(logits, dim=-1) # Use raw learned probabilities
            
            dist = torch.distributions.Categorical(probs)
            chosen_idx = dist.sample().item() # Sample naturally from the distribution!

        # Logging
        if hasattr(governor, "output_layer_history"):
            governor.output_layer_history.append(probs)
            
        return governor.strategies[chosen_idx].choose_action(observation, choices, wealths, raw_rewards, alive_mask)
    
    if is_learning_model:
        # Swap behavioral hook to use natural sampling
        governor.choose_action = native_sampling_choose_action
        
    test_metrics = run_phase(governor, test_n_trials, steps, n_agents, is_training=False)

    if is_learning_model:
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

    # Reset trackers to capture ONLY this visual test game cleanly
    if hasattr(governor, "input_layer_history"):
        governor.input_layer_history.clear()
    if hasattr(governor, "output_layer_history"):
        governor.output_layer_history.clear()
    if hasattr(governor, "reward_layer_history"):
        governor.reward_layer_history.clear()

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

    #plotting input layer dynamics
    plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends.png")
    #plotting output layer dynamics
    plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends.png")
    #plotting reward dynamics
    plot_reward_decomposition_isolated(governor, save_path="test_PPO/policy_reward_alignment.png")


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
    run_batch_simulation(train_n_trials=1000, test_n_trials=100, steps=1000)