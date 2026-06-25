import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import PPOGovernor, PigouvianGovernor
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.core.experiment_runner import ExperimentRunner


def plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends.png"):
    """Generates a comprehensive line graph tracking the governor input features."""
    if not hasattr(governor, "input_layer_history") or len(governor.input_layer_history) == 0:
        print("⚠️ No input layer history found to plot.")
        return

    history_matrix = np.array(governor.input_layer_history)
    if history_matrix.ndim != 2 or history_matrix.shape[1] < 1:
        print("⚠️ Input history data is malformed.")
        return

    timesteps = range(len(history_matrix))
    labels = [
        "Alive Ratio",
        "Mean Wealth",
        "Wealth Variance",
        "Min Wealth",
        *[f"Arm {i} Congestion" for i in range(history_matrix.shape[1] - 6)],
        "Max Congestion",
        "Action Diversity"
    ]

    plt.figure(figsize=(14, 7))
    for i in range(history_matrix.shape[1]):
        plt.plot(timesteps, history_matrix[:, i], label=labels[i], alpha=0.8, linewidth=1.5)

    plt.title("PPO Governor Global Observation Dynamics", fontsize=14, fontweight='bold')
    plt.xlabel("Decision Interval", fontsize=12)
    plt.ylabel("Normalized Feature Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Dashboard input tracking plot saved to: '{save_path}'")


def plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends.png"):
    """Plot the softmax strategy probabilities over decision intervals."""
    if not hasattr(governor, "output_layer_history") or len(governor.output_layer_history) == 0:
        print("⚠️ No output layer history found to plot.")
        return

    history_matrix = np.array(governor.output_layer_history)
    timesteps = range(len(history_matrix))
    labels = ["Pigouvian(δ=0.15)", "Progressive", "Wealth Multiplier", "FreeMarket" ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    plt.figure(figsize=(12, 6))
    for i in range(history_matrix.shape[1]):
        plt.plot(timesteps, history_matrix[:, i], label=labels[i], color=colors[i], linewidth=2)

    plt.title("PPO Governor Strategy Probabilities Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Decision Interval", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Strategy probability plot saved to: '{save_path}'")


def plot_reward_decomposition_isolated(governor, save_path="test_PPO/reward_decomposition.png"):
    """Creates a reward decomposition plot for the PPO governor's recorded intervals."""
    if not hasattr(governor, "reward_layer_history") or len(governor.reward_layer_history) == 0:
        print("⚠️ No reward history found for plotting.")
        return

    reward_matrix = np.array(governor.reward_layer_history)
    timesteps = range(len(reward_matrix))

    plt.figure(figsize=(12, 6.5))
    plt.plot(timesteps, reward_matrix[:, 0], color='#000000', label='Total Interval Reward', linewidth=3)
    plt.plot(timesteps, reward_matrix[:, 1], color='#9467bd', label='Economy Signal', linewidth=2, linestyle='--')
    plt.plot(timesteps, reward_matrix[:, 2], color='#8c564b', label='Ecology Signal', linewidth=2, linestyle='-.')
    plt.plot(timesteps, reward_matrix[:, 3], color='#e377c2', label='Equality Signal', linewidth=2, linestyle=':')

    plt.title("Reward Decomposition per PPO Decision Interval", fontsize=14, fontweight='bold')
    plt.xlabel("Decision Interval", fontsize=12)
    plt.ylabel("Reward Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Reward decomposition plot saved to: '{save_path}'")


def plot_actor_critic_diagnostics(diagnostics_history, save_dir="./test_PPO"):
    """Generates a 2x2 multi-panel diagnostic dashboard tracking the internal optimization health of the Actor and Critic networks.
    
    This function visualizes four key PPO training metrics across all update iterations:
    1. Critic value loss and explained variance (how well the baseline predicts returns)
    2. Policy entropy (exploration level in the action distribution)
    3. KL divergence (how far each PPO step moves from the old policy)
    4. Actor policy loss (optimization signal for the strategy selection head)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract diagnostic time series from the history array.
    trials = np.arange(1, len(diagnostics_history) + 1)
    v_loss = [d["value_loss"] for d in diagnostics_history]
    p_loss = [d["policy_loss"] for d in diagnostics_history]
    entropy = [d["entropy"] for d in diagnostics_history]
    kl_div = [d["kl_divergence"] for d in diagnostics_history]
    exp_var = [d["explained_variance"] for d in diagnostics_history]

    # Initialize the academic 2x2 grid style dashboard.
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

    # Instantiate a second y-axis sharing the same x-axis for Explained Variance.
    ax1_twin = ax1.twinx()
    color = 'tab:green'
    ax1_twin.set_ylabel('Explained Variance (Target: 1.0)', color=color, fontweight='bold')
    ax1_twin.plot(trials, exp_var, color=color, linestyle=":", alpha=0.9, linewidth=2, label="Explained Var")
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1_twin.axhline(0.0, color='black', linestyle='-', alpha=0.3)
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
    # Highlight a standard target safety roof line (typically around 0.015 - 0.03).
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

    # Clean layout wrapping up.
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(save_dir, "actor_critic_health.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Internal Actor-Critic structural dashboard saved to: '{save_path}'")


def run_phase(governor, n_trials, steps, n_agents, is_training=True, seed=42):
    """Run a block of independent trials and optionally train the PPO governor.

    During training, this function runs the environment for each trial,
    lets the governor collect decision-interval trajectories, and calls
    `update_ppo()` after each trial so the agent learns from the most recent
    10-step macro transitions.
    """
    batch_total_rewards = []
    batch_survival_counts = []
    batch_extinction_steps = []

    original_stdout = sys.stdout
    phase_name = "Training" if is_training else "Testing"

    for trial in range(n_trials):
        if hasattr(governor, "reset"):
            governor.reset()

        env = DepletingCommonsEnvironment(
            n_agents=n_agents,
            death_threshold=0.0,
            initial_wealth=250.0,
            step_cost=3.0,
            governor=governor,
            seed=seed
        )
        agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)

        try:
            # Silence verbose environment output during bulk training/testing.
            sys.stdout = open(os.devnull, 'w')
            runner.run(
                plot_rewards=False,
                plot_frequencies=False,
                plot_beliefs=False,
                plot_environment_health=False,
                plot_resource_efficiency=False,
            )
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        if is_training and hasattr(governor, "update_ppo"):
            if len(governor.buffer_states) > 0:
                governor.update_ppo()
            else:
                print(f"⚠️ Warning: No PPO transitions collected in trial {trial + 1}. Skipping update.")
        elif hasattr(governor, "buffer_states"):
            # In testing mode we clear any leftover PPO buffers.
            governor.buffer_states.clear()
            governor.buffer_actions.clear()
            governor.buffer_log_probs.clear()
            governor.buffer_values.clear()
            governor.buffer_rewards.clear()

        batch_total_rewards.append(sum(runner.total_rewards))
        survivor_count = sum(1 for step in env.death_steps if step is None)
        batch_survival_counts.append(survivor_count)

        if survivor_count == 0:
            extinction_step = max(step for step in env.death_steps if step is not None)
            batch_extinction_steps.append(extinction_step)

        if (trial + 1) % 10 == 0:
            print(f" -> [{phase_name}] Completed {trial + 1}/{n_trials} trials...")
        seed += 1  # Increment seed for next trial to ensure variability

    return {
        "avg_reward": np.mean(batch_total_rewards),
        "std_reward": np.std(batch_total_rewards),
        "max_reward": np.max(batch_total_rewards),
        "min_reward": np.min(batch_total_rewards),
        "avg_survivors": np.mean(batch_survival_counts),
        "extinction_rate": (len(batch_extinction_steps) / n_trials) * 100,
        "extinction_steps": batch_extinction_steps,
    }


def run_batch_simulation(train_n_trials=100, test_n_trials=100, steps=1000):
    """Build and execute the PPO governor training + evaluation workflow."""
    n_agents = 30
    seed = 200000

    governor = PPOGovernor(
            n_agents=n_agents,
            n_arms=30,
            decision_interval=10,
            actor_lr=3e-4,
            critic_lr=5e-4,
            clip_epsilon=0.2,
            ppo_epochs=3,
            batch_size=32,
            gamma=0.99,
            entropy_coef=0.0001,
            lam=0.95,
            seed=seed,
        )


    governor_name = governor.__class__.__name__
    is_learning_model = hasattr(governor, "update_ppo")

    print(f"🚀 Starting {governor_name} training run...")
    train_metrics = run_phase(governor, train_n_trials, steps, n_agents, is_training=True, seed=seed)

    # Plot internal PPO training diagnostics after training phase completes.
    if is_learning_model and hasattr(governor, "ppo_diagnostics_history"):
        print("\n📊 Generating Actor-Critic internal health diagnostic dashboard...")
        plot_actor_critic_diagnostics(diagnostics_history=governor.ppo_diagnostics_history, save_dir="./test_PPO")

    print(f"\n🧪 Evaluating {governor_name} performance...")
    # Evaluation uses argmax selection instead of sampling.
    governor.is_evaluating = True
    test_metrics = run_phase(governor, test_n_trials, steps, n_agents, is_training=False, seed=seed)
    governor.is_evaluating = False

    print("\n" + "=" * 60)
    print(f"📊 SYSTEM SUMMARY REPORT: {governor_name}")
    print("=" * 60)
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
    print("=" * 60)

    print("\n📈 Generating detailed internal dynamics visualizations...")
    plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends_all_runs.png")
    plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends_all_runs.png")
    plot_reward_decomposition_isolated(governor, save_path="test_PPO/policy_reward_alignment_all_runs.png")

    print("\n📸 Producing visualization trial outputs in './test_PPO'...")

    # Reset trackers to capture ONLY this visual test game cleanly
    if hasattr(governor, "input_layer_history"):
        governor.input_layer_history.clear()
    if hasattr(governor, "output_layer_history"):
        governor.output_layer_history.clear()
    if hasattr(governor, "reward_layer_history"):
        governor.reward_layer_history.clear()

    governor.reset()
    governor.is_evaluating = True

    visual_env = DepletingCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=0.0,
        initial_wealth=250.0,
        step_cost=3.0,
        governor=governor,
        seed=seed
    )
    visual_agents = [EpsilonGreedyAgent(visual_env.n_arms) for _ in range(n_agents)]
    visual_runner = ExperimentRunner(visual_env, visual_agents, timestep_limit=steps, save_dir="test_PPO")

    # Single visualization run to save plots and confirm behavior.
    visual_runner.run(
        plot_rewards=True,
        plot_frequencies=True,
        plot_beliefs=True,
        plot_environment_health=True,
        plot_resource_efficiency=True,
    )
    
    plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends_single_run.png")
    plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends_single_run.png")
    plot_reward_decomposition_isolated(governor, save_path="test_PPO/policy_reward_alignment_single_run.png")
    print("🎉 Visual diagnostic complete! Check './test_PPO' for charts.")
    


if __name__ == "__main__":
    run_batch_simulation(train_n_trials=1000, test_n_trials=100, steps=1000)