import json
from pathlib import Path

from multi_agent_bandits.core.governor import (
    FreeMarketGovernor,
    PigouvianGovernor,
    ProgressiveTaxGovernor,
    WealthMultiplierGovernor,
)
from multi_agent_bandits.experiments.plotting import (
    plot_critic_performance,
    plot_kl_divergence,
    plot_performance_comparison,
    plot_policy_entropy,
    plot_strategy_probabilities,
    plot_total_reward_curves,
)
from multi_agent_bandits.experiments.ppo_run_batch import run_batch_simulation as run_ppo_batch
from multi_agent_bandits.experiments.strat_run_batch import run_batch_simulation as run_baseline_batch

SEEDS = [100000, 200000, 300000, 400000, 500000]
#SEEDS = [123456]
#RESULTS_DIR = Path("results")
RESULTS_DIR = Path("results3")
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    baseline_definitions = [
        #("Free Market", FreeMarketGovernor(n_agents=30)),
        ("Pigouvian", PigouvianGovernor(n_agents=30, delta_drain=0.50, survival_threshold=100.0)),
        #("Progressive", ProgressiveTaxGovernor(n_agents=30, tax_rate=0.40)),
        #("Wealth Multiplier", WealthMultiplierGovernor(n_agents=30, base_tax_rate=0.016, max_tax_rate=0.08, subsidy_scale=2.0)),
    ]

    ppo_results = run_ppo_batch(train_n_trials=1, test_n_trials=1, steps=1000, seeds=SEEDS)

    baseline_results = {}
    for name, governor in baseline_definitions:
        baseline_results[name] = run_baseline_batch(
            test_n_trials=100,
            steps=1000,
            seeds=SEEDS,
            governor=governor,
        )

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
    plot_strategy_probabilities(
        ppo_results["strategy_probabilities"],
        save_path=str(PLOTS_DIR / "strategy_probabilities.png"),
        title="Strategy Probabilities (Training + Testing)",
    )
    plot_strategy_probabilities(
        ppo_results["test_strategy_probabilities"],
        save_path=str(PLOTS_DIR / "strategy_probabilities_testing.png"),
        title="Strategy Probabilities (1 Episode / 1000 Steps)",
        xlabel="Decision Interval",
    )

    comparison_data = {"PPO": ppo_results["eval_rewards"]}
    for name, _ in baseline_definitions:
        comparison_data[name] = [
            baseline_results[name]["seed_results"][seed]["avg_reward"]
            for seed in SEEDS
        ]

    plot_performance_comparison(
        comparison_data,
        save_path=str(PLOTS_DIR / "performance_comparison.png"),
    )
    plot_total_reward_curves(ppo_results["training_rewards"], save_path=str(PLOTS_DIR / "learning_curve.png"))

    print("\n=== Experiment Summary ===")
    print(f"PPO eval reward: {ppo_results['aggregate']['mean_eval_reward']:.2f} ± {ppo_results['aggregate']['std_eval_reward']:.2f}")
    for name, _ in baseline_definitions:
        print(
            f"{name} eval reward: {baseline_results[name]['aggregate']['avg_reward']:.2f} ± {baseline_results[name]['aggregate']['std_reward']:.2f}"
        )
    print(f"Saved results -> {RESULTS_DIR / 'all_results.json'}")
    print(f"Saved plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
