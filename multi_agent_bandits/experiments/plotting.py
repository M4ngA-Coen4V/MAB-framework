import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


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


def plot_critic_performance(critic_loss_data, explained_var_data, save_path=None):
    critic_mean, critic_std = _compute_mean_std(critic_loss_data)
    exp_mean, exp_std = _compute_mean_std(explained_var_data)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(critic_mean) + 1)
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


def plot_strategy_probabilities(all_seed_probs, save_path=None):
    labels = ["Pigouvian", "Progressive", "Wealth Multiplier", "Free Market"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    means = []
    stds = []
    for idx in range(len(labels)):
        values = [np.asarray(seed_probs)[:, idx] for seed_probs in all_seed_probs]
        mean, std = _compute_mean_std(values)
        means.append(mean)
        stds.append(std)

    x = np.arange(1, len(means[0]) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, label in enumerate(labels):
        ax.plot(x, means[idx], color=colors[idx], linewidth=2, label=label)
        ax.fill_between(x, means[idx] - stds[idx], means[idx] + stds[idx], color=colors[idx], alpha=0.15)

    ax.set_xlabel("Decision Interval")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
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
