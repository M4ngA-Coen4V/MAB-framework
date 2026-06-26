import os
import sys
from typing import Dict, List

import numpy as np

from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.governor import PPOGovernor
from multi_agent_bandits.experiments.plotting import (
    plot_critic_performance,
    plot_kl_divergence,
    plot_performance_comparison,
    plot_policy_entropy,
    plot_strategy_probabilities,
    plot_total_reward_curves,
)
from multi_agent_bandits.experiments.seed_manager import generate_seeds, set_seed
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def run_single_episode_trace(governor, steps, n_agents, seed=42):
    """Run one evaluation episode and return the logs needed for shared visualizations."""
    if hasattr(governor, "reset"):
        governor.reset()

    env = DepletingCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=0.0,
        initial_wealth=250.0,
        step_cost=3.0,
        governor=governor,
        seed=seed,
    )
    agents = [EpsilonGreedyAgent(env.n_arms, seed=seed + i) for i in range(n_agents)]
    runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)

    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        runner.run(
            plot_rewards=False,
            plot_frequencies=False,
            plot_beliefs=False,
            plot_environment_health=False,
            plot_resource_efficiency=False,
        )
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = original_stdout

    return {
        "strategy_probabilities": list(getattr(governor, "strategy_prob_history", [])),
        "choices_log": list(runner.choices_log),
        "rewards_log": list(runner.rewards_log),
        "values_log": list(runner.values_log),
        "death_steps": list(env.death_steps),
        "environmental_health_history": list(getattr(env, "environmental_health_history", [])),
        "n_arms": env.n_arms,
        "arm_means": [arm.mean for arm in env.arms],
    }


def run_phase(governor, n_trials, steps, n_agents, is_training=True, seed=42):
    """Run a block of independent trials and optionally train the PPO governor."""
    batch_total_rewards = []
    batch_survival_counts = []
    batch_extinction_steps = []
    trial_seeds = generate_seeds(seed, n_trials)

    original_stdout = sys.stdout
    phase_name = "Training" if is_training else "Testing"

    for trial in range(n_trials):
        trial_seed = trial_seeds[trial]
        if hasattr(governor, "reset"):
            governor.reset()

        env = DepletingCommonsEnvironment(
            n_agents=n_agents,
            death_threshold=0.0,
            initial_wealth=250.0,
            step_cost=3.0,
            governor=governor,
            seed=trial_seed,
        )
        agents = [EpsilonGreedyAgent(env.n_arms, seed=trial_seed + i) for i in range(n_agents)]
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)

        try:
            sys.stdout = open(os.devnull, "w")
            runner.run(
                plot_rewards=False,
                plot_frequencies=False,
                plot_beliefs=False,
                plot_environment_health=False,
                plot_resource_efficiency=False,
            )
        finally:
            try:
                sys.stdout.close()
            except Exception:
                pass
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

        batch_total_rewards.append(float(sum(runner.total_rewards)))
        survivor_count = sum(1 for step in env.death_steps if step is None)
        batch_survival_counts.append(survivor_count)

        if survivor_count == 0:
            extinction_step = max(step for step in env.death_steps if step is not None)
            batch_extinction_steps.append(extinction_step)

        if (trial + 1) % 10 == 0:
            print(f" -> [{phase_name}] Completed {trial + 1}/{n_trials} trials...")

    return {
        "avg_reward": float(np.mean(batch_total_rewards)),
        "std_reward": float(np.std(batch_total_rewards)),
        "max_reward": float(np.max(batch_total_rewards)),
        "min_reward": float(np.min(batch_total_rewards)),
        "avg_survivors": float(np.mean(batch_survival_counts)),
        "extinction_rate": float((len(batch_extinction_steps) / n_trials) * 100),
        "extinction_steps": batch_extinction_steps,
        "reward_history": batch_total_rewards,
    }


def run_batch_simulation(train_n_trials=100, test_n_trials=100, steps=1000, seeds=None):
    """Run a multi-seed PPO training and evaluation workflow."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]

    n_agents = 30
    all_results = {}
    all_training_rewards = []
    all_strategy_probs = []
    all_training_strategy_probs = []
    all_test_strategy_probs = []
    all_critic_loss = []
    all_entropy = []
    all_kl = []
    all_explained_var = []
    all_eval_rewards = []

    for master_seed in seeds:
        set_seed(master_seed)
        governor = PPOGovernor(
            n_agents=n_agents,
            n_arms=30,
            decision_interval=10,
            actor_lr=2e-4,
            critic_lr=5e-5,
            clip_epsilon=0.2,
            ppo_epochs=3,
            batch_size=32,
            gamma=0.99,
            entropy_coef=0.01,
            lam=0.95,
            seed=master_seed,
        )

        print(f"🚀 PPO training with seed {master_seed}...")
        train_metrics = run_phase(
            governor,
            train_n_trials,
            steps,
            n_agents,
            is_training=True,
            seed=master_seed,
        )
        training_strategy_probs = list(governor.strategy_prob_history)
        governor.strategy_prob_history = []

        governor.is_evaluating = True
        single_episode_test_trace = run_single_episode_trace(
            governor,
            steps,
            n_agents,
            seed=master_seed + 10000,
        )
        governor.strategy_prob_history = []
        test_metrics = run_phase(
            governor,
            test_n_trials,
            steps,
            n_agents,
            is_training=False,
            seed=master_seed + 10000,
        )
        governor.is_evaluating = False
        test_strategy_probs = single_episode_test_trace["strategy_probabilities"]
        combined_strategy_probs = training_strategy_probs + test_strategy_probs

        all_results[master_seed] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "training_rewards": train_metrics["reward_history"],
            "strategy_probabilities": combined_strategy_probs,
            "training_strategy_probabilities": training_strategy_probs,
            "test_strategy_probabilities": test_strategy_probs,
            "diagnostic_trace": single_episode_test_trace,
            "critic_loss_history": governor.critic_loss_history,
            "entropy_history": governor.entropy_history,
            "kl_history": governor.kl_history,
            "explained_variance_history": governor.explained_variance_history,
            "final_eval_reward": test_metrics["avg_reward"],
        }
        all_training_rewards.append(train_metrics["reward_history"])
        all_strategy_probs.append(combined_strategy_probs)
        all_training_strategy_probs.append(training_strategy_probs)
        all_test_strategy_probs.append(test_strategy_probs)
        all_critic_loss.append(governor.critic_loss_history)
        all_entropy.append(governor.entropy_history)
        all_kl.append(governor.kl_history)
        all_explained_var.append(governor.explained_variance_history)
        all_eval_rewards.append(test_metrics["avg_reward"])

    return {
        "seed_results": all_results,
        "aggregate": {
            "mean_eval_reward": float(np.mean(all_eval_rewards)),
            "std_eval_reward": float(np.std(all_eval_rewards, ddof=0)),
            "mean_training_reward": float(np.mean([np.mean(values) for values in all_training_rewards])),
        },
        "training_rewards": all_training_rewards,
        "strategy_probabilities": all_strategy_probs,
        "training_strategy_probabilities": all_training_strategy_probs,
        "test_strategy_probabilities": all_test_strategy_probs,
        "diagnostic_traces": [result["diagnostic_trace"] for result in all_results.values()],
        "critic_loss_history": all_critic_loss,
        "entropy_history": all_entropy,
        "kl_history": all_kl,
        "explained_variance_history": all_explained_var,
        "eval_rewards": all_eval_rewards,
    }


if __name__ == "__main__":
    run_batch_simulation(train_n_trials=100, test_n_trials=100, steps=1000)

    print("🎉 Visual diagnostic complete! Check './test_PPO' for charts.")
    


if __name__ == "__main__":
    run_batch_simulation(train_n_trials=1000, test_n_trials=100, steps=1000)
