import argparse
import csv
import os
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.congested_commons_env import CongestedCommonsEnvironment, DepletingCommonsEnvironment
from multi_agent_bandits.core.governor import LearningGovernorAI, CommunistGovernor, SocialistGovernor
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent





def main(
        steps=1000, 
        save_dir=None, 
        plot_rewards=False, 
        plot_frequencies=False, 
        plot_beliefs=False, 
        plot_environment_health=False, 
        plot_resource_efficiency=False,
        death_threshold=0.0, 
        initial_wealth=200.0, 
        step_cost=3.0
        ):
    """Run the congested commons baseline experiment.

    Args:
        steps (int): number of timesteps to simulate.
        save_dir (str|None): directory to save logs (choices/rewards/metadata).
        plot_rewards (bool): whether to show cumulative reward plots.
        plot_frequencies (bool): whether to show arm frequency plots.
        plot_beliefs (bool): whether to show agent belief plots.
        plot_environment_health (bool): whether to show environment health plots.
        plot_resource_efficiency (bool): whether to show resource efficiency plots.
        death_threshold (float): wealth threshold below which agents die.
        initial_wealth (float): starting wealth for each agent.
        step_cost (float): wealth cost paid by each alive agent every timestep.
    """

    # Number of agents in the baseline scenario
    n_agents = 3

    # Instantiate the governor and the congested commons environment.
    governor = CommunistGovernor(n_agents=n_agents)
    #governor = None
    #governor = LearningGovernorAI(n_agents=n_agents)
    #governor = SocialistGovernor(n_agents=n_agents)

    # This uses default arms means [10, 2, 1] unless 'arms' is provided.
    env = CongestedCommonsEnvironment(
        n_agents=n_agents,
        death_threshold=death_threshold,
        initial_wealth=initial_wealth,
        step_cost=step_cost,
        governor=governor,
    )

    #env = DepletingCommonsEnvironment(
    #    n_agents=n_agents,
    #    death_threshold=death_threshold,
    #    initial_wealth=initial_wealth,
    #    step_cost=step_cost,
    #    governor=governor
    #    )


    # Create agents (default: epsilon-greedy). Agents will learn from rewards
    # delivered by the environment via their `update()` method.
    agents = [EpsilonGreedyAgent(env.n_arms) for _ in range(n_agents)]

    # Create the experiment runner to manage simulation loop and logging.
    runner = ExperimentRunner(env, agents, timestep_limit=steps, save_dir=save_dir)

    # Run the simulation. This will call env.step() repeatedly and update agents.
    runner.run(
        plot_rewards=plot_rewards, 
        plot_frequencies=plot_frequencies, 
        plot_beliefs=plot_beliefs, 
        plot_environment_health=plot_environment_health,
        plot_resource_efficiency=plot_resource_efficiency)




