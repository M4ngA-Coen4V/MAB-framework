import argparse
import csv
import os
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.congested_commons_env import CongestedCommonsEnvironment
from multi_agent_bandits.core.governor import LearningGovernorAI
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def _save_governor_history_csv(governor, output_dir):
    """Save a CSV file containing all governor debug steps."""
    if not hasattr(governor, "history") or not governor.history:
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "governor_history.csv")

    with open(file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "observation", "raw_adjustments", "adjusted", "reward", "death_count"])

        for step_idx, step in enumerate(governor.history, start=1):
            writer.writerow([
                step_idx,
                repr(step["observation"]),
                repr(step["raw_adjustments"]),
                repr(step["adjustments"]),
                f"{step['reward']:.6f}",
                step["death_count"],
            ])


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False, death_threshold=0.0, initial_wealth=200.0, step_cost=2.5):
    """Run the congested commons baseline experiment.

    Args:
        steps (int): number of timesteps to simulate.
        save_dir (str|None): directory to save logs (choices/rewards/metadata).
        plot_rewards (bool): whether to show cumulative reward plots.
        plot_frequencies (bool): whether to show arm frequency plots.
        death_threshold (float): wealth threshold below which agents die.
        initial_wealth (float): starting wealth for each agent.
        step_cost (float): wealth cost paid by each alive agent every timestep.
    """

    # Number of agents in the baseline scenario
    n_agents = 3

    # Instantiate the governor and the congested commons environment.
    governor = LearningGovernorAI(n_agents=n_agents)

    # This uses default arms means [10, 2, 1] unless 'arms' is provided.
    env = CongestedCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=death_threshold,
        initial_wealth=initial_wealth,
        step_cost=step_cost,
        governor=governor,
    )

    # Create agents (default: epsilon-greedy). Agents will learn from rewards
    # delivered by the environment via their `update()` method.
    agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]

    # Create the experiment runner to manage simulation loop and logging.
    runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=save_dir)

    # Run the simulation. This will call env.step() repeatedly and update agents.
    choices_log, rewards_log = runner.run(plot_rewards=plot_rewards, plot_frequencies=plot_frequencies)

    # Print runner's summary (total rewards per agent)
    runner.print_summary()

    # Additionally print environment-maintained wealth and alive status
    print("Agent wealths and alive status:")
    for i, (wealth, alive) in enumerate(zip(env.agent_wealths, env.is_alive)):
        status = "alive" if alive else "dead"
        print(f"  Agent {i} - wealth: {wealth:.2f} - {status}")

    # Calculate and print total social wealth
    total_combined_wealth = sum(env.agent_wealths)
    print("----------------------------")
    print(f"Total Combined Wealth (All Agents): {total_combined_wealth:.2f}")

    # Calculate and print total rewards across all agents
    total_combined_reward = sum(runner.total_rewards)
    print("----------------------------")
    print(f"Total Combined Reward (All Agents): {total_combined_reward:.2f}")

    # Print governor debug summary data if available
    if hasattr(governor, "history") and governor.history:
        print("----------------------------")
        print("Governor debug summary:")
        print(f"  Policy means: {[f'{m:.4f}' for m in governor.means]}")
        print(f"  Total governor steps recorded: {len(governor.history)}")

        first_steps = governor.history[:5]
        for step_idx, step in enumerate(first_steps, start=1):
            print(f"  Step {step_idx}: raw={step['raw_adjustments']}, adjusted={step['adjustments']}, reward={step['reward']:.2f}, deaths={step['death_count']}")

        avg_adjustments = [sum(step['adjustments'][i] for step in governor.history) / len(governor.history)
                           for i in range(n_agents)]
        print(f"  Avg governor adjustments: {[f'{a:.4f}' for a in avg_adjustments]}")

        output_dir = save_dir or os.path.join("results", "example")
        _save_governor_history_csv(governor, output_dir)
        print(f"Saved governor history CSV to: {os.path.join(output_dir, 'governor_history.csv')}")

    # Print each agent's survival duration based on death step tracking
    print("Agent survival duration:")
    for i, death_step in enumerate(env.death_steps):
        if death_step is None:
            print(f"  Agent {i} survived all {steps} steps")
        else:
            print(f"  Agent {i} died at step {death_step}")




