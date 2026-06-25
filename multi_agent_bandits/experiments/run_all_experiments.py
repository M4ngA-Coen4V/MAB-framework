import json
import os
from pathlib import Path

import numpy as np

from multi_agent_bandits.core.governor import FreeMarketGovernor, PigouvianGovernor
from multi_agent_bandits.experiments.ppo_run_batch import run_batch_simulation as run_ppo_batch
from multi_agent_bandits.experiments.plotting import (
    plot_critic_performance,
    plot_kl_divergence,
    plot_performance_comparison,
    plot_policy_entropy,
    plot_strategy_probabilities,
    plot_total_reward_curves,
)
from multi_agent_bandits.experiments.seed_manager import set_seed
from multi_agent_bandits.experiments.strat_run_batch import run_batch_simulation as run_baseline_batch

#SEEDS = [42, 123, 456, 789, 101112]
SEEDS = [42]
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    governor = PigouvianGovernor(n_agents=30, delta_drain=0.15, survival_threshold=100.0)

    ppo_results = run_ppo_batch(train_n_trials=400, test_n_trials=100, steps=1000, seeds=SEEDS)
    baseline_results = run_baseline_batch(test_n_trials=100, steps=1000, seeds=SEEDS, governor=governor)
    results_payload = {
        "seeds": SEEDS,
        "ppo": ppo_results,
        "baseline": baseline_results,
    }

    with open(RESULTS_DIR / "all_results.json", "w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    plot_critic_performance(
        ppo_results["critic_loss_history"],
        ppo_results["explained_variance_history"],
        save_path=str(PLOTS_DIR / "critic_performance.png"),
    )
    plot_policy_entropy(ppo_results["entropy_history"], save_path=str(PLOTS_DIR / "policy_entropy.png"))
    plot_kl_divergence(ppo_results["kl_history"], save_path=str(PLOTS_DIR / "kl_divergence.png"))
    plot_strategy_probabilities(ppo_results["strategy_probabilities"], save_path=str(PLOTS_DIR / "strategy_probabilities.png"))
    plot_performance_comparison(
        {
            "PPO": ppo_results["eval_rewards"],
            "Pigouvian": [baseline_results["seed_results"][seed]["avg_reward"] for seed in SEEDS],
            "Progressive": [baseline_results["seed_results"][seed]["avg_reward"] for seed in SEEDS],
            "Wealth Multiplier": [baseline_results["seed_results"][seed]["avg_reward"] for seed in SEEDS],
            "Free Market": [baseline_results["seed_results"][seed]["avg_reward"] for seed in SEEDS],
        },
        save_path=str(PLOTS_DIR / "performance_comparison.png"),
    )
    plot_total_reward_curves(ppo_results["training_rewards"], save_path=str(PLOTS_DIR / "learning_curve.png"))

    print("\n=== Experiment Summary ===")
    print(f"PPO eval reward: {ppo_results['aggregate']['mean_eval_reward']:.2f} ± {ppo_results['aggregate']['std_eval_reward']:.2f}")
    print(f"Baseline eval reward: {baseline_results['aggregate']['avg_reward']:.2f} ± {baseline_results['aggregate']['std_reward']:.2f}")
    print(f"Saved results -> {RESULTS_DIR / 'all_results.json'}")
    print(f"Saved plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
