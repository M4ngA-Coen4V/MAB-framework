import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _as_seed_arrays(data, expected_length=None):
    arrays = []
    for item in data:
        arr = np.asarray(item, dtype=float)
        if expected_length is not None and arr.ndim != 1:
            arr = arr.reshape(-1)
        arrays.append(arr)
    return arrays


def _compute_mean_std(arrays):
    arrays = [np.asarray(arr, dtype=float) for arr in arrays]
    if not arrays:
        return np.array([]), np.array([])
    stacked = np.vstack(arrays)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def _extract_final_belief_matrix(trace):
    values_log = trace.get("values_log", [])
    death_steps = trace.get("death_steps", [])
    n_arms = int(trace.get("n_arms", 0))
    n_agents = len(death_steps)

    if not values_log or n_agents == 0 or n_arms == 0:
        return np.zeros((3, n_arms), dtype=float)

    final_beliefs = np.zeros((n_agents, n_arms), dtype=float)
    last_step_index = len(values_log) - 1

    for agent_idx in range(n_agents):
        death_step = death_steps[agent_idx] if agent_idx < len(death_steps) else None
        if death_step is not None and death_step > 0 and death_step - 1 < len(values_log):
            final_beliefs[agent_idx, :] = np.asarray(values_log[death_step - 1][agent_idx], dtype=float)
        else:
            final_beliefs[agent_idx, :] = np.asarray(values_log[last_step_index][agent_idx], dtype=float)

    elite_indices = []
    struggling_indices = []
    starved_indices = []
    for idx, death_step in enumerate(death_steps):
        if death_step is None:
            elite_indices.append(idx)
        elif death_step > 150:
            struggling_indices.append(idx)
        else:
            starved_indices.append(idx)

    class_matrix = np.zeros((3, n_arms), dtype=float)
    class_matrix[0, :] = np.mean(final_beliefs[elite_indices, :], axis=0) if elite_indices else 0.0
    class_matrix[1, :] = np.mean(final_beliefs[struggling_indices, :], axis=0) if struggling_indices else 0.0
    class_matrix[2, :] = np.mean(final_beliefs[starved_indices, :], axis=0) if starved_indices else 0.0
    return class_matrix


def plot_agent_beliefs(runs, save_path=None, title="Socio-Economic Belief Mapping: Averaged Final Perceived Arm Values"):
    if not runs:
        print("No agent belief traces available to plot.")
        return None

    class_matrices = [_extract_final_belief_matrix(trace) for trace in runs]
    averaged_matrix = np.mean(np.stack(class_matrices, axis=0), axis=0)
    n_arms = averaged_matrix.shape[1]

    class_labels = [
        "Elite Survivors",
        "Struggling Mid-Class",
        "Early Starved Class",
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        averaged_matrix,
        cmap="viridis",
        xticklabels=max(1, n_arms // 10),
        yticklabels=class_labels,
        annot=False,
        cbar_kws={"label": "Average Perceived Reward Value"},
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Arm Index (Sorted Best to Worst Actual Base Mean)")
    ax.set_ylabel("Agent Survival Class")
    plt.yticks(rotation=0)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _extract_resource_efficiency_components(trace):
    health_history = np.asarray(trace.get("environmental_health_history", []), dtype=float)
    if health_history.size == 0:
        return None

    if health_history.ndim != 2:
        return None

    n_arms = int(trace.get("n_arms", health_history.shape[1]))
    timesteps = health_history.shape[0]
    base_means = np.asarray(trace.get("arm_means", []), dtype=float)
    if base_means.size == 0:
        base_means = np.zeros(n_arms, dtype=float)

    global_optimal_constant = float(np.sum(base_means))
    theoretical_ceiling = np.full(timesteps, global_optimal_constant, dtype=float)

    death_steps = trace.get("death_steps", [])
    n_agents = len(death_steps)
    alive_history = np.zeros((timesteps, n_agents), dtype=bool)
    for i, death_step in enumerate(death_steps):
        if death_step is None:
            alive_history[:, i] = True
        else:
            alive_history[:death_step, i] = True

    choices_log = trace.get("choices_log", [])
    rewards_log = trace.get("rewards_log", [])

    net_extracted_wealth = np.zeros(timesteps, dtype=float)
    environmental_degradation_loss = np.zeros(timesteps, dtype=float)
    opportunity_cost_loss = np.zeros(timesteps, dtype=float)

    for t in range(timesteps):
        if np.sum(alive_history[t, :]) == 0:
            opportunity_cost_loss[t] = global_optimal_constant
            continue

        if len(rewards_log) > t:
            step_rewards = np.asarray(rewards_log[t], dtype=float)
            living_rewards = step_rewards[alive_history[t, :]]
            net_extracted_wealth[t] = float(np.sum(living_rewards))

        if len(choices_log) > t:
            step_choices = np.asarray(choices_log[t])
            living_choices = step_choices[alive_history[t, :]]
            unique_chosen_arms = np.unique(living_choices).astype(int)
            unchosen_arms = np.setdiff1d(np.arange(n_arms), unique_chosen_arms).astype(int)

            opportunity_cost_loss[t] = float(np.sum(base_means[unchosen_arms]))

            kappa_t = health_history[t, :]
            chosen_arms_capacity_loss = base_means[unique_chosen_arms] * (1.0 - kappa_t[unique_chosen_arms])
            environmental_degradation_loss[t] = float(np.sum(chosen_arms_capacity_loss))

            calculated_total = net_extracted_wealth[t] + opportunity_cost_loss[t] + environmental_degradation_loss[t]
            if calculated_total > global_optimal_constant:
                opportunity_cost_loss[t] = max(0.0, global_optimal_constant - net_extracted_wealth[t] - environmental_degradation_loss[t])

    active_population = np.sum(alive_history, axis=1)
    return {
        "net_extracted_wealth": net_extracted_wealth,
        "environmental_degradation_loss": environmental_degradation_loss,
        "opportunity_cost_loss": opportunity_cost_loss,
        "theoretical_ceiling": theoretical_ceiling,
        "active_population": active_population,
    }


def plot_resource_efficiency(runs, save_path=None, title="Macroscopic Resource Utilization & Decentralized Waste Breakdown"):
    if not runs:
        print("No resource efficiency traces available to plot.")
        return None

    components = [_extract_resource_efficiency_components(trace) for trace in runs]
    components = [component for component in components if component is not None]
    if not components:
        print("No valid resource efficiency traces available to plot.")
        return None

    net_extracted_wealth = np.mean([component["net_extracted_wealth"] for component in components], axis=0)
    environmental_degradation_loss = np.mean([component["environmental_degradation_loss"] for component in components], axis=0)
    opportunity_cost_loss = np.mean([component["opportunity_cost_loss"] for component in components], axis=0)
    theoretical_ceiling = np.mean([component["theoretical_ceiling"] for component in components], axis=0)
    active_population = np.mean([component["active_population"] for component in components], axis=0)

    timesteps = len(net_extracted_wealth)
    steps = np.arange(timesteps) + 1

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.stackplot(
        steps,
        net_extracted_wealth,
        environmental_degradation_loss,
        opportunity_cost_loss,
        labels=[
            "Net System Wealth Extracted (Blue)",
            "Value Lost to Environmental Degradation (Orange)",
            "Value Lost to Leftover Opportunity Cost (Red)",
        ],
        colors=["#2b8cbe", "#fe9929", "#e34a33"],
        alpha=0.85,
    )
    ax1.plot(steps, theoretical_ceiling, label="Theoretical Pareto-Optimal Capacity Ceiling", color="black", linestyle="--", linewidth=2.5)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Simulation Timestep", fontsize=11)
    ax1.set_ylabel("Total Resource Value (Units of Reward)", fontsize=11)
    ax1.set_xlim(0, timesteps)
    ax1.set_ylim(0, float(np.max(theoretical_ceiling)) * 1.05 if len(theoretical_ceiling) else 1.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(steps, active_population, color="#555555", linestyle=":", linewidth=2, label="Active Living Population")
    ax2.set_ylabel("Active Population Count", color="#555555", fontsize=11)
    ax2.set_ylim(0, max(1.0, float(np.max(active_population)) * 1.05))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10, frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_critic_performance(critic_loss_data, explained_var_data, save_path=None):
    critic_mean, critic_std = _compute_mean_std(critic_loss_data)
    exp_mean, exp_std = _compute_mean_std(explained_var_data)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(critic_mean) + 1)
    ax1.set_title("Critic Performance Evaluation", fontsize=12, fontweight="bold")
    ax1.plot(x, critic_mean, color="tab:red", linewidth=2, label="Critic MSE Loss")
    ax1.fill_between(x, critic_mean - critic_std, critic_mean + critic_std, color="tab:red", alpha=0.2)
    ax1.set_xlabel("Training Update")
    ax1.set_ylabel("Critic MSE Value Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, exp_mean, color="tab:blue", linewidth=2, label="Explained Variance")
    ax2.fill_between(x, exp_mean - exp_std, exp_mean + exp_std, color="tab:blue", alpha=0.15)
    ax2.set_ylabel("Explained Variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_policy_entropy(entropy_data, save_path=None):
    mean, std = _compute_mean_std(entropy_data)
    x = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Actor Exploration / Policy Randomness", fontsize=12, fontweight="bold")
    ax.plot(x, mean, color="purple", linewidth=2, label="Entropy")
    ax.fill_between(x, mean - std, mean + std, color="purple", alpha=0.2)
    ax.set_xlabel("Training Update")
    ax.set_ylabel("Shannon Entropy")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_kl_divergence(kl_data, target_ceiling=0.02, save_path=None):
    mean, std = _compute_mean_std(kl_data)
    x = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Policy Update Shift Distance (KL)", fontsize=12, fontweight="bold")
    ax.plot(x, mean, color="darkorange", linewidth=2, label="Approx. KL")
    ax.fill_between(x, mean - std, mean + std, color="darkorange", alpha=0.2)
    ax.axhline(target_ceiling, color="red", linestyle="--", label="Target Ceiling")
    ax.set_xlabel("Training Update")
    ax.set_ylabel("Approximate KL Divergence")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_strategy_probabilities(all_seed_probs, save_path=None, title="Strategy Probabilities", xlabel="Decision Interval"):
    labels = ["Pigouvian", "Progressive", "Wealth Multiplier", "Free Market"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    means = []
    stds = []
    for idx in range(len(labels)):
        values = []
        for seed_probs in all_seed_probs:
            if not seed_probs:
                values.append(np.array([]))
                continue
            arr = np.asarray(seed_probs, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            values.append(arr[:, idx])
        mean, std = _compute_mean_std(values)
        means.append(mean)
        stds.append(std)

    x = np.arange(1, len(means[0]) + 1) if means else np.array([])
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, label in enumerate(labels):
        ax.plot(x, means[idx], color=colors[idx], linewidth=2, label=label)
        ax.fill_between(x, means[idx] - stds[idx], means[idx] + stds[idx], color=colors[idx], alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_performance_comparison(results_dict, save_path=None):
    conditions = ["PPO", "Pigouvian", "Progressive", "Wealth Multiplier", "Free Market"]
    means = []
    stds = []
    for condition in conditions:
        if condition not in results_dict:
            continue
        values = results_dict[condition]
        values = np.asarray(values, dtype=float)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=0)))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(conditions))
    ax.set_title("Performance Comparison Across Governors", fontsize=12, fontweight="bold")
    ax.bar(x, means, yerr=stds, capsize=4, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20)
    ax.set_ylabel("Mean Evaluation Reward")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_total_reward_curves(all_seed_rewards, save_path=None):
    mean, std = _compute_mean_std(all_seed_rewards)
    x = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("PPO Governor Learning Curve", fontsize=12, fontweight="bold")
    ax.plot(x, mean, color="tab:green", linewidth=2, label="Mean Total Reward")
    ax.fill_between(x, mean - std, mean + std, color="tab:green", alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total System Reward")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
