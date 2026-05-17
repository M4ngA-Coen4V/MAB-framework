import argparse
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.congested_commons_env import CongestedCommonsEnvironment
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False, death_threshold=0.0, initial_wealth=0.0):
    """Run the congested commons baseline experiment.

    Args:
        steps (int): number of timesteps to simulate.
        save_dir (str|None): directory to save logs (choices/rewards/metadata).
        plot_rewards (bool): whether to show cumulative reward plots.
        plot_frequencies (bool): whether to show arm frequency plots.
        death_threshold (float): wealth threshold below which agents die.
        initial_wealth (float): starting wealth for each agent.
    """

    # Number of agents in the baseline scenario
    n_agents = 3

    # Instantiate the congested commons environment with defaults.
    # This uses default arms means [10, 2, 1] unless 'arms' is provided.
    env = CongestedCommonsEnvironment(n_agents=n_agents, death_threshold=death_threshold, initial_wealth=initial_wealth)

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

