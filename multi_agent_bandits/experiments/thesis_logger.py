import json
import math
from pathlib import Path

import numpy as np


WEALTH_STRATEGY_INDEX = 2


def _safe_mean(values):
    values = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values):
    values = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    if not values:
        return float("nan")
    return float(np.std(values, ddof=0))


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _checkpoint_values(histories, checkpoints):
    if not histories:
        return []
    output = []
    for checkpoint in checkpoints:
        values = []
        for history in histories:
            history = np.asarray(history, dtype=float)
            if history.ndim == 0 or history.size == 0:
                continue
            index = min(int(checkpoint), history.shape[0] - 1)
            values.append(float(history[index]))
        output.append(_safe_mean(values))
    return output


def _extract_final_beliefs(values_log, death_steps):
    if not values_log or not death_steps:
        return np.array([])

    n_agents = len(death_steps)
    n_arms = len(values_log[-1][0]) if values_log and values_log[-1] else 0
    if n_arms == 0:
        return np.array([])

    final_beliefs = np.zeros((n_agents, n_arms), dtype=float)
    last_step_index = len(values_log) - 1
    for agent_idx in range(n_agents):
        death_step = death_steps[agent_idx]
        if death_step is not None and death_step > 0 and death_step - 1 < len(values_log):
            final_beliefs[agent_idx, :] = np.asarray(values_log[death_step - 1][agent_idx], dtype=float)
        else:
            final_beliefs[agent_idx, :] = np.asarray(values_log[last_step_index][agent_idx], dtype=float)
    return final_beliefs


def summarize_episode_trace(*, env, runner, strategy_probabilities, decision_interval, arm_means, seed=None, phase=None):
    """Create a compact per-episode summary for thesis reporting."""
    strategy_probabilities = np.asarray(strategy_probabilities, dtype=float)
    if strategy_probabilities.ndim == 1 and strategy_probabilities.size:
        strategy_probabilities = strategy_probabilities.reshape(1, -1)

    wealth_history = np.asarray(getattr(env, "wealth_history", []), dtype=float)
    health_history = np.asarray(getattr(env, "environmental_health_history", []), dtype=float)
    death_steps = list(getattr(env, "death_steps", []))
    n_agents = len(death_steps)
    n_arms = len(arm_means)

    if strategy_probabilities.size:
        wealth_probs = strategy_probabilities[:, WEALTH_STRATEGY_INDEX]
        switch_candidates = np.where(wealth_probs < 0.5)[0]
        switch_interval = int(switch_candidates[0] + 1) if len(switch_candidates) else int(strategy_probabilities.shape[0])
    else:
        switch_interval = 0

    switch_step = min(max(switch_interval * decision_interval, 1), len(wealth_history) if len(wealth_history) else 1)
    switch_index = max(0, switch_step - 1)

    if len(health_history):
        switch_health = np.asarray(health_history[min(switch_index, len(health_history) - 1)], dtype=float)
        if switch_health.size != n_arms:
            switch_health = np.ones(n_arms, dtype=float)
    else:
        switch_health = np.ones(n_arms, dtype=float)

    resource_wealth_at_switch = float(np.sum(np.asarray(arm_means, dtype=float) * switch_health))

    if len(wealth_history):
        wealth_snapshot = np.asarray(wealth_history[min(switch_index, len(wealth_history) - 1)], dtype=float)
    else:
        wealth_snapshot = np.zeros(n_agents, dtype=float)

    alive_mask_at_switch = [death_step is None or death_step > switch_step for death_step in death_steps]
    surviving_agents_at_switch = int(sum(alive_mask_at_switch))

    if wealth_snapshot.size:
        agent_wealth_mean_at_switch = float(np.mean(wealth_snapshot))
        agent_wealth_variance_at_switch = float(np.var(wealth_snapshot, ddof=0))
    else:
        agent_wealth_mean_at_switch = float("nan")
        agent_wealth_variance_at_switch = float("nan")

    # Resource efficiency components, averaged over timesteps within the episode.
    net_extracted = []
    degradation = []
    opportunity_cost = []

    choices_log = getattr(runner, "choices_log", [])
    rewards_log = getattr(runner, "rewards_log", [])

    if len(health_history):
        alive_history = np.zeros((len(health_history), n_agents), dtype=bool)
        for agent_idx, death_step in enumerate(death_steps):
            if death_step is None:
                alive_history[:, agent_idx] = True
            else:
                alive_history[:death_step, agent_idx] = True

        for t in range(len(health_history)):
            if np.sum(alive_history[t, :]) == 0:
                net_extracted.append(0.0)
                degradation.append(0.0)
                opportunity_cost.append(float(np.sum(arm_means)))
                continue

            step_rewards = np.asarray(rewards_log[t], dtype=float) if len(rewards_log) > t else np.zeros(n_agents, dtype=float)
            living_rewards = step_rewards[alive_history[t, :]]
            net_extracted.append(float(np.sum(living_rewards)))

            step_choices = np.asarray(choices_log[t]) if len(choices_log) > t else np.array([])
            living_choices = step_choices[alive_history[t, :]] if step_choices.size else np.array([])
            unique_chosen_arms = np.unique(living_choices).astype(int) if living_choices.size else np.array([], dtype=int)
            unchosen_arms = np.setdiff1d(np.arange(n_arms), unique_chosen_arms).astype(int)

            opportunity_cost.append(float(np.sum(np.asarray(arm_means)[unchosen_arms])))

            kappa_t = np.asarray(health_history[t], dtype=float)
            if unique_chosen_arms.size:
                chosen_arms_capacity_loss = np.asarray(arm_means)[unique_chosen_arms] * (1.0 - kappa_t[unique_chosen_arms])
                degradation.append(float(np.sum(chosen_arms_capacity_loss)))
            else:
                degradation.append(0.0)
    else:
        net_extracted = [float(sum(step)) for step in rewards_log]
        degradation = [0.0 for _ in rewards_log]
        opportunity_cost = [0.0 for _ in rewards_log]

    final_beliefs = _extract_final_beliefs(getattr(runner, "values_log", []), death_steps)
    elite_indices = [idx for idx, death_step in enumerate(death_steps) if death_step is None]
    mid_indices = [idx for idx, death_step in enumerate(death_steps) if death_step is not None and death_step > 150]
    starved_indices = [idx for idx, death_step in enumerate(death_steps) if death_step is not None and death_step <= 150]

    if elite_indices and final_beliefs.size:
        elite_beliefs_mean = np.mean(final_beliefs[elite_indices, :], axis=0)
        elite_pearson_corr = _safe_corr(np.asarray(arm_means, dtype=float), elite_beliefs_mean)
    else:
        elite_pearson_corr = float("nan")

    return {
        "seed": seed,
        "phase": phase,
        "decision_interval": int(decision_interval),
        "episode_steps": int(len(rewards_log)),
        "strategy_probabilities": np.asarray(strategy_probabilities, dtype=float).tolist(),
        "switch_interval": int(switch_interval),
        "switch_step": int(switch_step),
        "resource_wealth_at_switch": resource_wealth_at_switch,
        "surviving_agents_at_switch": surviving_agents_at_switch,
        "agent_wealth_mean_at_switch": agent_wealth_mean_at_switch,
        "agent_wealth_variance_at_switch": agent_wealth_variance_at_switch,
        "net_extracted_wealth_total": float(np.sum(net_extracted)),
        "environmental_degradation_loss_total": float(np.sum(degradation)),
        "opportunity_cost_loss_total": float(np.sum(opportunity_cost)),
        "net_extracted_wealth_mean": float(np.mean(net_extracted)) if net_extracted else float("nan"),
        "environmental_degradation_loss_mean": float(np.mean(degradation)) if degradation else float("nan"),
        "opportunity_cost_loss_mean": float(np.mean(opportunity_cost)) if opportunity_cost else float("nan"),
        "mean_active_population": float(np.mean([np.sum([death_step is None or death_step > (t + 1) for death_step in death_steps]) for t in range(len(rewards_log))])) if rewards_log else float("nan"),
        "elite_survivors_count": int(len(elite_indices)),
        "mid_class_count": int(len(mid_indices)),
        "early_starved_count": int(len(starved_indices)),
        "elite_pearson_corr": elite_pearson_corr,
        "final_total_reward": float(sum(getattr(runner, "total_rewards", []))),
        "n_agents": int(n_agents),
        "n_arms": int(n_arms),
        "arm_means": [float(value) for value in arm_means],
    }


def write_thesis_report(*, results_dir, ppo_results, baseline_results, seeds, evaluation_episodes_per_seed, steps_per_episode):
    """Write a JSON payload and human-readable markdown summary for the thesis results chapter."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ppo_seed_results = ppo_results["seed_results"]
    baseline_seed_results = {name: result["seed_results"] for name, result in baseline_results.items()}

    def _seed_eval_reward(seed_result):
        if "avg_reward" in seed_result:
            return float(seed_result["avg_reward"])
        if "final_eval_reward" in seed_result:
            return float(seed_result["final_eval_reward"])
        if "test_metrics" in seed_result and isinstance(seed_result["test_metrics"], dict):
            return float(seed_result["test_metrics"].get("avg_reward", float("nan")))
        return float("nan")

    ppo_eval_rewards = [_seed_eval_reward(ppo_seed_results[seed]) for seed in seeds]
    baseline_eval_rewards = {
        name: [_seed_eval_reward(seed_results[seed]) for seed in seeds]
        for name, seed_results in baseline_seed_results.items()
    }

    ppo_episode_summaries = []
    for seed in seeds:
        ppo_episode_summaries.extend(ppo_seed_results[seed].get("evaluation_episode_summaries", []))

    baseline_episode_summaries = {}
    for name in baseline_results:
        episode_summaries = []
        for seed in seeds:
            episode_summaries.extend(baseline_seed_results[name][seed].get("evaluation_episode_summaries", []))
        baseline_episode_summaries[name] = episode_summaries

    def _average_strategy_probabilities(episode_summaries):
        if not episode_summaries:
            return [], []
        arrays = [np.asarray(summary["strategy_probabilities"], dtype=float) for summary in episode_summaries if summary.get("strategy_probabilities")]
        if not arrays:
            return [], []
        stacked = np.stack(arrays, axis=0)
        return stacked.mean(axis=0).tolist(), stacked.std(axis=0, ddof=0).tolist()

    def _average_dict_metrics(episode_summaries, keys):
        averages = {}
        for key in keys:
            values = [summary.get(key) for summary in episode_summaries if summary.get(key) is not None]
            averages[key] = {
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }
        return averages

    ppo_strategy_mean, ppo_strategy_std = _average_strategy_probabilities(ppo_episode_summaries)
    ppo_switch_stats = _average_dict_metrics(ppo_episode_summaries, ["switch_interval", "resource_wealth_at_switch", "surviving_agents_at_switch", "agent_wealth_mean_at_switch", "agent_wealth_variance_at_switch"])
    ppo_mechanism_corr = _average_dict_metrics(ppo_episode_summaries, ["elite_pearson_corr"])

    rq2_condition_metrics = {}
    for name in ["PPO", "Pigouvian", "Free Market", "Wealth Multiplier", "Progressive"]:
        if name == "PPO":
            episodes = ppo_episode_summaries
            seed_rewards = ppo_eval_rewards
        else:
            episodes = baseline_episode_summaries[name]
            seed_rewards = baseline_eval_rewards[name]
        rq2_condition_metrics[name] = {
            "seed_rewards": [float(value) for value in seed_rewards],
            "reward_mean": _safe_mean(seed_rewards),
            "reward_std": _safe_std(seed_rewards),
            "net_extracted_wealth_mean": _safe_mean([episode.get("net_extracted_wealth_mean") for episode in episodes]),
            "environmental_degradation_loss_mean": _safe_mean([episode.get("environmental_degradation_loss_mean") for episode in episodes]),
            "opportunity_cost_loss_mean": _safe_mean([episode.get("opportunity_cost_loss_mean") for episode in episodes]),
            "elite_survivors_mean": _safe_mean([episode.get("elite_survivors_count") for episode in episodes]),
            "mid_class_mean": _safe_mean([episode.get("mid_class_count") for episode in episodes]),
            "early_starved_mean": _safe_mean([episode.get("early_starved_count") for episode in episodes]),
        }

    ppo_reward_threshold_49000 = {
        "all_above_49000": all(value > 49000 for value in ppo_eval_rewards),
        "seed_rewards": ppo_eval_rewards,
    }

    # Theoretical ceiling is per-step capacity; multiply by episode steps for a comparable cumulative ceiling.
    arm_means = np.asarray(ppo_episode_summaries[0]["arm_means"] if ppo_episode_summaries else [], dtype=float)
    theoretical_episode_ceiling = float(np.sum(arm_means) * steps_per_episode) if arm_means.size else float("nan")

    report_payload = {
        "metadata": {
            "seeds": list(seeds),
            "evaluation_episodes_per_seed": int(evaluation_episodes_per_seed),
            "steps_per_episode": int(steps_per_episode),
            "decision_intervals_per_episode": int(steps_per_episode // 10),
        },
        "raw_episode_summaries": {
            "PPO": ppo_episode_summaries,
            "Pigouvian": baseline_episode_summaries["Pigouvian"],
            "Free Market": baseline_episode_summaries["Free Market"],
            "Wealth Multiplier": baseline_episode_summaries["Wealth Multiplier"],
            "Progressive": baseline_episode_summaries["Progressive"],
        },
        "rq1": {
            "strategy_probabilities_mean": ppo_strategy_mean,
            "strategy_probabilities_std": ppo_strategy_std,
            "average_switch_interval": _safe_mean([episode.get("switch_interval") for episode in ppo_episode_summaries]),
            "switch_interval_std": _safe_std([episode.get("switch_interval") for episode in ppo_episode_summaries]),
            "switch_statistics": ppo_switch_stats,
            "environment_state_at_switch": {
                "resource_wealth_at_switch": ppo_switch_stats["resource_wealth_at_switch"],
                "surviving_agents_at_switch": ppo_switch_stats["surviving_agents_at_switch"],
                "agent_wealth_mean_at_switch": ppo_switch_stats["agent_wealth_mean_at_switch"],
                "agent_wealth_variance_at_switch": ppo_switch_stats["agent_wealth_variance_at_switch"],
            },
        },
        "rq2": {
            "per_seed_rewards": {
                "PPO": ppo_eval_rewards,
                "Pigouvian": baseline_eval_rewards["Pigouvian"],
                "Free Market": baseline_eval_rewards["Free Market"],
                "Wealth Multiplier": baseline_eval_rewards["Wealth Multiplier"],
                "Progressive": baseline_eval_rewards["Progressive"],
            },
            "per_seed_reward_arrays": {
                "PPO": ppo_eval_rewards,
                "Pigouvian": baseline_eval_rewards["Pigouvian"],
                "Free Market": baseline_eval_rewards["Free Market"],
                "Wealth Multiplier": baseline_eval_rewards["Wealth Multiplier"],
                "Progressive": baseline_eval_rewards["Progressive"],
            },
            "conditions": rq2_condition_metrics,
            "theoretical_episode_ceiling": theoretical_episode_ceiling,
            "reward_unit_explanation": "Episode rewards are summed across all agents and all 1000 steps; the per-step resource ceiling is the sum of arm means, so the comparable episode ceiling is sum(arm means) * 1000.",
        },
        "rq3": {
            "ppo_training_curves": {
                "mean_eval_reward": ppo_results["aggregate"]["mean_eval_reward"],
                "std_eval_reward": ppo_results["aggregate"]["std_eval_reward"],
                "critic_loss_history": {
                    "mean": np.mean(np.asarray(ppo_results["critic_loss_history"], dtype=float), axis=0).tolist() if ppo_results["critic_loss_history"] else [],
                    "std": np.std(np.asarray(ppo_results["critic_loss_history"], dtype=float), axis=0, ddof=0).tolist() if ppo_results["critic_loss_history"] else [],
                },
                "entropy_history": {
                    "mean": np.mean(np.asarray(ppo_results["entropy_history"], dtype=float), axis=0).tolist() if ppo_results["entropy_history"] else [],
                    "std": np.std(np.asarray(ppo_results["entropy_history"], dtype=float), axis=0, ddof=0).tolist() if ppo_results["entropy_history"] else [],
                },
                "kl_history": {
                    "mean": np.mean(np.asarray(ppo_results["kl_history"], dtype=float), axis=0).tolist() if ppo_results["kl_history"] else [],
                    "std": np.std(np.asarray(ppo_results["kl_history"], dtype=float), axis=0, ddof=0).tolist() if ppo_results["kl_history"] else [],
                },
                "explained_variance_history": {
                    "mean": np.mean(np.asarray(ppo_results["explained_variance_history"], dtype=float), axis=0).tolist() if ppo_results["explained_variance_history"] else [],
                    "std": np.std(np.asarray(ppo_results["explained_variance_history"], dtype=float), axis=0, ddof=0).tolist() if ppo_results["explained_variance_history"] else [],
                },
            },
            "raw_seed_histories": {
                "training_rewards": ppo_results.get("training_rewards", []),
                "strategy_probabilities": ppo_results.get("strategy_probabilities", []),
                "training_strategy_probabilities": ppo_results.get("training_strategy_probabilities", []),
                "test_strategy_probabilities": ppo_results.get("test_strategy_probabilities", []),
                "critic_loss_history": ppo_results.get("critic_loss_history", []),
                "entropy_history": ppo_results.get("entropy_history", []),
                "kl_history": ppo_results.get("kl_history", []),
                "explained_variance_history": ppo_results.get("explained_variance_history", []),
            },
            "checkpoint_summary": {
                "reward": _checkpoint_values(ppo_results.get("training_rewards", []), [0, 500, 999]),
                "critic_loss": _checkpoint_values(ppo_results.get("critic_loss_history", []), [0, 500, 999]),
                "explained_variance": _checkpoint_values(ppo_results.get("explained_variance_history", []), [0, 500, 999]),
                "entropy": _checkpoint_values(ppo_results.get("entropy_history", []), [0, 500, 999]),
                "kl_divergence": _checkpoint_values(ppo_results.get("kl_history", []), [0, 500, 999]),
            },
            "seed_rewards": ppo_eval_rewards,
            "all_seeds_above_49000": ppo_reward_threshold_49000["all_above_49000"],
        },
        "mechanism": {
            "elite_survivor_correlations": {
                "PPO": [episode.get("elite_pearson_corr") for episode in ppo_episode_summaries],
                "Pigouvian": [episode.get("elite_pearson_corr") for episode in baseline_episode_summaries["Pigouvian"]],
                "Wealth Multiplier": [episode.get("elite_pearson_corr") for episode in baseline_episode_summaries["Wealth Multiplier"]],
            },
            "elite_survivor_correlation_summary": {
                "PPO": _average_dict_metrics(ppo_episode_summaries, ["elite_pearson_corr"]),
                "Pigouvian": _average_dict_metrics(baseline_episode_summaries["Pigouvian"], ["elite_pearson_corr"]),
                "Wealth Multiplier": _average_dict_metrics(baseline_episode_summaries["Wealth Multiplier"], ["elite_pearson_corr"]),
            },
        },
    }

    json_path = results_dir / "thesis_statistics.json"
    md_path = results_dir / "thesis_statistics.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    lines = []
    lines.append("# Thesis Results Summary")
    lines.append("")
    lines.append(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    lines.append(f"Evaluation episodes per seed: {evaluation_episodes_per_seed}")
    lines.append(f"Steps per episode: {steps_per_episode}")
    lines.append("")
    lines.append("## RQ1 Dynamic Behavior")
    lines.append(f"Average switch interval: {_safe_mean([episode.get('switch_interval') for episode in ppo_episode_summaries]):.2f}")
    lines.append(f"Std dev of switch interval: {_safe_std([episode.get('switch_interval') for episode in ppo_episode_summaries]):.2f}")
    lines.append(f"Resource wealth at switch (mean): {ppo_switch_stats['resource_wealth_at_switch']['mean']:.2f}")
    lines.append(f"Surviving agents at switch (mean): {ppo_switch_stats['surviving_agents_at_switch']['mean']:.2f}")
    lines.append(f"Switch point array: {[episode.get('switch_interval') for episode in ppo_episode_summaries[:10]]}")
    lines.append("")
    lines.append("## RQ2 Performance vs Baselines")
    for name, stats in rq2_condition_metrics.items():
        lines.append(f"- {name}: rewards={stats['seed_rewards']}")
        lines.append(f"  extracted={stats['net_extracted_wealth_mean']:.2f}, degradation={stats['environmental_degradation_loss_mean']:.2f}, oppCost={stats['opportunity_cost_loss_mean']:.2f}")
    lines.append(f"Theoretical episode ceiling: {theoretical_episode_ceiling:.2f}")
    lines.append("")
    lines.append("## RQ3 Learning Stability")
    lines.append(f"All PPO seeds above 49000: {ppo_reward_threshold_49000['all_above_49000']}")
    lines.append(f"Checkpoint reward values: {_checkpoint_values(ppo_results.get('training_rewards', []), [0, 500, 999])}")
    lines.append(f"Checkpoint critic loss values: {_checkpoint_values(ppo_results.get('critic_loss_history', []), [0, 500, 999])}")
    lines.append(f"Checkpoint explained variance values: {_checkpoint_values(ppo_results.get('explained_variance_history', []), [0, 500, 999])}")
    lines.append(f"Checkpoint entropy values: {_checkpoint_values(ppo_results.get('entropy_history', []), [0, 500, 999])}")
    lines.append(f"Checkpoint KL values: {_checkpoint_values(ppo_results.get('kl_history', []), [0, 500, 999])}")
    lines.append("")
    lines.append("## Mechanism")
    lines.append(f"PPO elite survivor correlation (mean): {ppo_mechanism_corr['elite_pearson_corr']['mean']:.4f}")
    lines.append(f"Pigouvian elite survivor correlation (mean): {_average_dict_metrics(baseline_episode_summaries['Pigouvian'], ['elite_pearson_corr'])['elite_pearson_corr']['mean']:.4f}")
    lines.append(f"Wealth Multiplier elite survivor correlation (mean): {_average_dict_metrics(baseline_episode_summaries['Wealth Multiplier'], ['elite_pearson_corr'])['elite_pearson_corr']['mean']:.4f}")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_payload