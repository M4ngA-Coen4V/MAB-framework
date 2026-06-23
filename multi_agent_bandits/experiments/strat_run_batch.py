import os
import sys
import numpy as np
from multi_agent_bandits.core.congested_commons_env import DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import PigouvianGovernor, ProgressiveTaxGovernor, SurvivalTargetedGovernor, FreeMarketGovernor, WealthMultiplierGovernor
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.core.experiment_runner import ExperimentRunner


def run_phase(governor, n_trials, steps, n_agents):
    """Run evaluation-only trials for a fixed, non-learning governor.

    This function runs `n_trials` independent episodes using `governor` (which
    must implement the same interface as the governors in this repo). It does
    NOT perform any training or weight updates; it only collects episode
    statistics and lets the ExperimentRunner save its own visualization plots.
    """
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
            governor=governor
        )

        agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]
        runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=None)

        try:
            # Silence verbose environment output during batch evaluation
            sys.stdout = open(os.devnull, 'w')
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

        if (trial + 1) % 10 == 0:
            print(f" -> [Evaluation] Completed {trial + 1}/{n_trials} trials...")

    return {
        "avg_reward": np.mean(batch_total_rewards),
        "std_reward": np.std(batch_total_rewards),
        "max_reward": np.max(batch_total_rewards),
        "min_reward": np.min(batch_total_rewards),
        "avg_survivors": np.mean(batch_survival_counts),
        "extinction_rate": (len(batch_extinction_steps) / n_trials) * 100,
        "extinction_steps": batch_extinction_steps,
    }


def run_batch_simulation(test_n_trials=100, steps=1000):
    """Run a batch evaluation for a single (non-learning) governor class.
    This runner does NOT perform any training.
    """
    n_agents=30

    #governor = PigouvianGovernor(n_agents=n_agents, delta_drain=0.30, survival_threshold=20.0)
    #governor = ProgressiveTaxGovernor(n_agents=n_agents, tax_rate=0.10)
    governor = FreeMarketGovernor(n_agents=n_agents)
    #governor = WealthMultiplierGovernor(n_agents=n_agents, base_tax_rate=0.016, max_tax_rate=0.08, subsidy_scale=2.0)
    governor_name = governor.__class__.__name__

    print(f"🧪 Evaluating static strategy: {governor_name} over {test_n_trials} trials...")
    test_metrics = run_phase(governor, test_n_trials, steps, n_agents)

    print("\n" + "=" * 60)
    print(f"📊 SYSTEM SUMMARY REPORT: {governor_name}")
    print("=" * 60)
    print(f"| Metric                      | Testing Phase       |")
    print(f"|-----------------------------|---------------------|")
    print(f"| Avg Combined System Reward  | {test_metrics['avg_reward']:.2f} ± {test_metrics['std_reward']:.2f}  |")
    print(f"| Max Combined System Reward  | {test_metrics['max_reward']:.2f}            |")
    print(f"| Min Combined System Reward  | {test_metrics['min_reward']:.2f}            |")
    print(f"| Avg Active Survivors / {n_agents}   | {test_metrics['avg_survivors']:.1f}                  |")
    print(f"| Complete Extinction Rate    | {test_metrics['extinction_rate']:.1f}%                |")
    if test_metrics['extinction_steps']:
        e_ext = f"{np.mean(test_metrics['extinction_steps']):.1f}"
        print(f"| Avg Extinction Timestep     | {e_ext:<19} |")
    print("=" * 60)

    # Visual diagnostic single run: let ExperimentRunner save its own plots
    print("\n📸 Producing single visual trial outputs in './test_strategy'...")
    if hasattr(governor, "reset"):
        governor.reset()

    visual_env = DepletingCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=0.0,
        initial_wealth=250.0,
        step_cost=3.0,
        governor=governor,
    )
    visual_agents = [EpsilonGreedyAgent(visual_env.n_arms) for _ in range(n_agents)]
    visual_runner = ExperimentRunner(visual_env, visual_agents, timestep_limit=steps, save_dir="test_strategy")

    # Run with standard plotting enabled so ExperimentRunner writes plots
    visual_runner.run(
        plot_rewards=True,
        plot_frequencies=True,
        plot_beliefs=True,
        plot_environment_health=True,
        plot_resource_efficiency=True,
    )

    print("🎉 Visual diagnostic complete! Check './test_strategy' for charts.")


if __name__ == "__main__":
    # Default: evaluate PigouvianGovernor with parameters matching the PPO baseline's strategy.
    run_batch_simulation(test_n_trials=10, steps=1000)