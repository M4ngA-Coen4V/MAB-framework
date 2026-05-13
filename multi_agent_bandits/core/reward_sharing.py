def linear_share(raw_reward, n_agents):
    """Equal split (default in the environment)."""
    return [raw_reward / n_agents] * n_agents

def zero_on_collision(raw_reward, n_agents):
    """Everyone gets zero in case of collision."""
    return [0.0] * n_agents

def winner_takes_all(raw_reward, n_agents):
    """Pick one agent randomly to get the whole reward"""
    import random
    rewards = [0.0] * n_agents
    winner = random.randrange(n_agents)
    rewards[winner] = raw_reward
    return rewards

def custom_100_20_10_share(raw_reward, n_agents):
    """First agent gets 100%, second gets 20%, third gets 10% (if there are that many agents)."""
    if n_agents == 1:
        return [raw_reward]
    elif n_agents == 2:
        return [raw_reward * 0.2] * 2
    elif n_agents >= 3:
        return [raw_reward * 0.1] * 3
    else:
        raise ValueError("Invalid number of agents: {}".format(n_agents))
    