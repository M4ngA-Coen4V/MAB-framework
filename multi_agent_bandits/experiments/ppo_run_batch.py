import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import PPOGovernor
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
    labels = ["Pigouvian", "Progressive", "Survival", "FreeMarket"]
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


def run_phase(governor, n_trials, steps, n_agents, is_training=True):
    """Execute multiple environment trials and collect aggregate metrics."""
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
            initial_wealth=200.0,
            step_cost=3.0,
            governor=governor
        )
        agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)

        try:
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
    n_agents = 30
    governor = PPOGovernor(
        n_agents=n_agents,
        n_arms=30,
        decision_interval=10,
        learning_rate=1e-3,
        clip_epsilon=0.2,
        ppo_epochs=4,
        batch_size=32,
        gamma=0.99,
        entropy_coef=0.01,
        seed=None,
    )

    governor_name = governor.__class__.__name__
    is_learning_model = hasattr(governor, "update_ppo")

    print(f"🚀 Starting {governor_name} training run...")
    train_metrics = run_phase(governor, train_n_trials, steps, n_agents, is_training=True)

    print(f"\n🧪 Evaluating {governor_name} performance...")
    governor.is_evaluating = True
    test_metrics = run_phase(governor, test_n_trials, steps, n_agents, is_training=False)
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

    print("\n📸 Producing visualization trial outputs in './test_PPO'...")
    governor.reset()
    governor.is_evaluating = True

    visual_env = DepletingCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=0.0,
        initial_wealth=200.0,
        step_cost=3.0,
        governor=governor,
    )
    visual_agents = [EpsilonGreedyAgent(visual_env.n_arms) for _ in range(n_agents)]
    visual_runner = ExperimentRunner(visual_env, visual_agents, timestep_limit=steps, save_dir="test_PPO")
    visual_runner.run(
        plot_rewards=True,
        plot_frequencies=True,
        plot_beliefs=True,
        plot_environment_health=True,
        plot_resource_efficiency=True,
    )
    plot_input_layer_dynamics(governor, save_path="test_PPO/input_layer_trends.png")
    plot_output_layer_dynamics(governor, save_path="test_PPO/output_layer_trends.png")
    plot_reward_decomposition_isolated(governor, save_path="test_PPO/policy_reward_alignment.png")

    print("🎉 Visual diagnostic complete! Check './test_PPO' for charts.")


if __name__ == "__main__":
    run_batch_simulation(train_n_trials=100, test_n_trials=100, steps=1000)
