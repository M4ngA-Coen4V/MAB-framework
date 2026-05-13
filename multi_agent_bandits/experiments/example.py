from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.reward_sharing import linear_share, zero_on_collision, winner_takes_all

from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent


def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):

    n_agents = 3

    arms = [
        Arm(mean=1.0, sd=1.0),
        Arm(mean=2.0, sd=1.0),
        Arm(mean=1.5, sd=1.0)
    ]

    env = Environment(
        n_agents=n_agents,
        arms=arms,
        collision_policy=zero_on_collision
    )

    agents = [
        RandomAgent(env.n_arms),
        EpsilonGreedyAgent(env.n_arms),
        UCB_BaselineAgent(env.n_arms)
    ]

    runner = ExperimentRunner(
        env,
        agents,
        timestep_limit=steps,
        save_dir=save_dir
    )

    runner.run(
        plot_rewards=plot_rewards,
        plot_frequencies=plot_frequencies
    )

    # testing git
    runner.print_summary() 
