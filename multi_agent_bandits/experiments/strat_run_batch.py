import os
import sys
from typing import Dict, List

import numpy as np

from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.governor import (
    FreeMarketGovernor,
    PigouvianGovernor,
    ProgressiveTaxGovernor,
    SurvivalTargetedGovernor,
    WealthEqualizerGovernor,
    WealthMultiplierGovernor,
)
from multi_agent_bandits.experiments.seed_manager import set_seed
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def run_phase(governor, n_trials, steps, n_agents, seed=42):
    """Run evaluation-only trials for a fixed, non-learning governor."""
    batch_total_rewards = []
    batch_survival_counts = []
    batch_extinction_steps = []

    original_stdout = sys.stdout

    for trial in range(n_trials):
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

        batch_total_rewards.append(sum(runner.total_rewards))
        survivor_count = sum(1 for step in env.death_steps if step is None)
        batch_survival_counts.append(survivor_count)

        if survivor_count == 0:
            extinction_step = max(step for step in env.death_steps if step is not None)
            batch_extinction_steps.append(extinction_step)

    return {
        "avg_reward": float(np.mean(batch_total_rewards)),
        "std_reward": float(np.std(batch_total_rewards)),
        "max_reward": float(np.max(batch_total_rewards)),
        "min_reward": float(np.min(batch_total_rewards)),
        "avg_survivors": float(np.mean(batch_survival_counts)),
        "extinction_rate": float((len(batch_extinction_steps) / n_trials) * 100),
        "extinction_steps": batch_extinction_steps,
    }


def run_batch_simulation(test_n_trials=100, steps=1000, seeds=None, governor=None):
    """Run a multi-seed evaluation for one or more non-learning governors."""
    if seeds is None:
        seeds = [43, 123, 456, 789, 101112]

    n_agents = 30
    if governor is None:
        #governor = FreeMarketGovernor(n_agents=n_agents)
        governor = PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=100.0)
    governor_name = governor.__class__.__name__

    seed_results = {}
    for seed in seeds:
        set_seed(seed)
        # Rebuild a fresh governor instance with seed for each run.
        if governor_name == "PigouvianGovernor":
            governor_seeded = PigouvianGovernor(n_agents=n_agents, delta_drain=0.15, survival_threshold=100.0)
        elif governor_name == "ProgressiveTaxGovernor":
            governor_seeded = ProgressiveTaxGovernor(n_agents=n_agents, tax_rate=0.40)
        elif governor_name == "SurvivalTargetedGovernor":
            governor_seeded = SurvivalTargetedGovernor(n_agents=n_agents, survival_cost=3.0)
        elif governor_name == "WealthMultiplierGovernor":
            governor_seeded = WealthMultiplierGovernor(n_agents=n_agents, base_tax_rate=0.016, max_tax_rate=0.08, subsidy_scale=2.0)
        elif governor_name == "WealthEqualizerGovernor":
            governor_seeded = WealthEqualizerGovernor(n_agents=n_agents, base_tax_rate=0.002, max_tax_rate=0.01, subsidy_scale=0.0)
        else:
            governor_seeded = FreeMarketGovernor(n_agents=n_agents)

        print(f"🧪 Evaluating {governor_name} with seed {seed} over {test_n_trials} trials...")
        metrics = run_phase(governor_seeded, test_n_trials, steps, n_agents, seed=seed)
        seed_results[seed] = metrics

    aggregate = {
        "avg_reward": float(np.mean([result["avg_reward"] for result in seed_results.values()])),
        "std_reward": float(np.std([result["avg_reward"] for result in seed_results.values()], ddof=0)),
        "max_reward": float(np.max([result["max_reward"] for result in seed_results.values()])),
        "min_reward": float(np.min([result["min_reward"] for result in seed_results.values()])),
    }

    print("\n" + "=" * 60)
    print(f"📊 SYSTEM SUMMARY REPORT: {governor_name}")
    print("=" * 60)
    print(f"| Metric                      | Mean ± Std         |")
    print(f"|-----------------------------|--------------------|")
    print(f"| Avg Combined System Reward  | {aggregate['avg_reward']:.2f} ± {aggregate['std_reward']:.2f} |")
    print(f"| Max Combined System Reward  | {aggregate['max_reward']:.2f} |")
    print(f"| Min Combined System Reward  | {aggregate['min_reward']:.2f} |")
    print("=" * 60)

    return {"seed_results": seed_results, "aggregate": aggregate}


if __name__ == "__main__":
    run_batch_simulation(test_n_trials=100, steps=1000)