import json
from pathlib import Path

from multi_agent_bandits.core.governor import (
    FreeMarketGovernor,
    PigouvianGovernor,
    ProgressiveTaxGovernor,
    WealthMultiplierGovernor,
)
from multi_agent_bandits.experiments.plotting import (
    plot_agent_beliefs,
    plot_critic_performance,
    plot_kl_divergence,
    plot_performance_comparison,
    plot_policy_entropy,
    plot_resource_efficiency,
    plot_strategy_probabilities,
    plot_total_reward_curves,
)
from multi_agent_bandits.experiments.ppo_run_batch import run_batch_simulation as run_ppo_batch
from multi_agent_bandits.experiments.strat_run_batch import run_batch_simulation as run_baseline_batch
from multi_agent_bandits.experiments.thesis_logger import write_thesis_report


def _safe_name(name):
    return name.lower().replace(" ", "_")

SEEDS = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000, 3000000]
#SEEDS = [123456]
#RESULTS_DIR = Path("results")
RESULTS_DIR = Path("results6")
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    baseline_definitions = [
        ("Free Market", FreeMarketGovernor(n_agents=30)),
        ("Pigouvian", PigouvianGovernor(n_agents=30, delta_drain=0.15, survival_threshold=100.0)),
        ("Progressive", ProgressiveTaxGovernor(n_agents=30, tax_rate=0.40)),
        ("Wealth Multiplier", WealthMultiplierGovernor(n_agents=30, base_tax_rate=0.016, max_tax_rate=0.08, subsidy_scale=2.0)),
    ]

    ppo_results = run_ppo_batch(train_n_trials=9, test_n_trials=3, steps=1000, seeds=SEEDS)

    baseline_results = {}
    for name, governor in baseline_definitions:
        baseline_results[name] = run_baseline_batch(
            test_n_trials=3,
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

    write_thesis_report(
        results_dir=RESULTS_DIR,
        ppo_results=ppo_results,
        baseline_results=baseline_results,
        seeds=SEEDS,
        evaluation_episodes_per_seed=100,
        steps_per_episode=1000,
    )

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

    plot_agent_beliefs(
        ppo_results["diagnostic_traces"],
        save_path=str(PLOTS_DIR / "ppo_agent_beliefs.png"),
        title="PPO Governor: Averaged Agent Beliefs",
    )
    plot_resource_efficiency(
        ppo_results["diagnostic_traces"],
        save_path=str(PLOTS_DIR / "ppo_resource_efficiency.png"),
        title="PPO Governor: Averaged Resource Efficiency",
    )

    for name, _ in baseline_definitions:
        plot_agent_beliefs(
            baseline_results[name]["diagnostic_traces"],
            save_path=str(PLOTS_DIR / f"{_safe_name(name)}_agent_beliefs.png"),
            title=f"{name}: Averaged Agent Beliefs",
        )
        plot_resource_efficiency(
            baseline_results[name]["diagnostic_traces"],
            save_path=str(PLOTS_DIR / f"{_safe_name(name)}_resource_efficiency.png"),
            title=f"{name}: Averaged Resource Efficiency",
        )

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
