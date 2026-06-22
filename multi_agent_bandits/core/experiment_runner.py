import os
import csv
import matplotlib.pyplot as plt

class ExperimentRunner:
    def __init__(self, env, agents, timestep_limit=1000, save_dir=None):
        self.env = env
        self.agents = agents
        self.T = timestep_limit
        self.choices_log = []
        self.rewards_log = []
        self.values_log = []
        self.total_rewards = [0.0] * len(agents)
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def print_experiment_info(self):
        print("==== Experiment Setup ====")
        print(f"Timesteps: {self.T}")
        print(f"Number of arms: {self.env.n_arms}")
        print("Arms:")
        for i, arm in enumerate(self.env.arms):
            print(f"  Arm {i}: mean={arm.mean}, sd={arm.sd}")
        print(f"Collision policy: {self.env.collision_policy.__name__}")
        print(f"Governor: {self.env.governor.__class__.__name__}")
        print(f"Death threshold: {self.env.death_threshold}")
        print(f"Initial wealth: {self.env.initial_wealth}")
        print(f"Step cost: {self.env.step_cost}")
        print(f"Number of agents: {len(self.agents)}")
        print("Agents:")
        for i, ag in enumerate(self.agents):
            print(f"  - Agent {i}: {ag.name}")
        print("==========================")

    def run(self, plot_rewards=False, plot_frequencies=False, plot_beliefs=False, plot_environment_health=False, plot_resource_efficiency=False):
        self.print_experiment_info()

        for t in range(self.T):
            choices, rewards = self.env.step(self.agents)
            self.choices_log.append(choices)
            self.rewards_log.append(rewards)
            for i, r in enumerate(rewards):
                self.total_rewards[i] += r
            
            # Extract a snapshot copy of self.values for every agent at this step
            # Iterate through the elements of the list
            step_values_snapshot = []
            for agent in self.agents:
                # Access the 'values' attribute belonging to the individual agent object
                raw_values = [float(v) for v in agent.values]
                step_values_snapshot.append(raw_values)
                
            self.values_log.append(step_values_snapshot)

        if self.save_dir:
            self.save_logs()
            self._save_governor_history_csv()

        # Handle runtime analytical summaries
        self.print_summary()

        if plot_rewards:
            self.plot_reward_trajectories()

        if plot_frequencies:
            self.plot_arm_frequencies()

        if plot_beliefs:
            self.plot_agent_beliefs()
        
        if plot_environment_health:
            self.plot_environmental_health()
        
        if plot_resource_efficiency:
            self.plot_resource_efficiency()

        return self.choices_log, self.rewards_log

    def save_logs(self):
        choices_path = os.path.join(self.save_dir, "choices.csv")
        with open(choices_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"agent_{i}" for i in range(len(self.agents))])
            writer.writerows(self.choices_log)

        rewards_path = os.path.join(self.save_dir, "rewards.csv")
        with open(rewards_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"agent_{i}" for i in range(len(self.agents))])
            writer.writerows(self.rewards_log)

        metadata_path = os.path.join(self.save_dir, "metadata.csv")
        with open(metadata_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["timesteps", self.T])
            writer.writerow(["n_arms", self.env.n_arms])
            for i, arm in enumerate(self.env.arms):
                writer.writerow([f"arm_{i}_mean", arm.mean])
                writer.writerow([f"arm_{i}_sd", arm.sd])
            writer.writerow(["collision_policy", self.env.collision_policy.__name__])
            writer.writerow(["governor", self.env.governor.__class__.__name__])
            writer.writerow(["death_threshold", self.env.death_threshold])
            writer.writerow(["initial_wealth", self.env.initial_wealth])
            writer.writerow(["step_cost", self.env.step_cost])
            writer.writerow(["n_agents", len(self.agents)])
            for i, ag in enumerate(self.agents):
                writer.writerow([f"agent_{i}_name", ag.name])

        summary_path = os.path.join(self.save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("==== Experiment Summary ====\n")
            f.write(f"Timesteps: {self.T}\n")
            f.write(f"Number of arms: {self.env.n_arms}\n")
            f.write("Arms:\n")
            for i, arm in enumerate(self.env.arms):
                f.write(f"  Arm {i}: mean={arm.mean}, sd={arm.sd}\n")
            f.write(f"Collision policy: {self.env.collision_policy.__name__}\n")
            f.write(f"Governor: {self.env.governor.__class__.__name__}\n")
            f.write(f"Death threshold: {self.env.death_threshold}\n")
            f.write(f"Initial wealth: {self.env.initial_wealth}\n")
            f.write(f"Step cost: {self.env.step_cost}\n")
            f.write(f"Number of agents: {len(self.agents)}\n")
            f.write("Agents:\n")
            for i, ag in enumerate(self.agents):
                f.write(f"  - Agent {i}: {ag.name}\n")
            f.write("Total rewards:\n")
            for i, total in enumerate(self.total_rewards):
                f.write(f"  Agent {i} ({self.agents[i].name}): {total:.2f}\n")
            f.write("============================\n")
            total_combined_wealth = sum(self.env.agent_wealths)
            f.write(f"Total Combined Wealth (All Agents): {total_combined_wealth:.2f}\n")
            f.write("============================\n")
            total_combined_reward = sum(self.total_rewards)
            f.write(f"Total Combined Reward (All Agents): {total_combined_reward:.2f}\n")
            f.write("============================\n")
            f.write("Agent survival duration:\n")
            for i, death_step in enumerate(self.env.death_steps):
                if death_step is None:
                    f.write(f"  Agent {i} survived all {self.T} steps\n")
                else:
                    f.write(f"  Agent {i} died at step {death_step}\n")

            print("----------------------------")
        print("Agent survival duration:")
        for i, death_step in enumerate(self.env.death_steps):
            if death_step is None:
                print(f"  Agent {i} survived all {self.T} steps")
            else:
                print(f"  Agent {i} died at step {death_step}")
        print("============================")

    def _save_governor_history_csv(self):
        """Save a CSV file containing all governor debug steps directly to save_dir."""
        gov = self.env.governor
        if not gov or not hasattr(gov, "history") or not gov.history:
            return

        file_path = os.path.join(self.save_dir, "governor_history.csv")
        with open(file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "observation", "raw_adjustments", "adjusted", "reward", "death_count"])

            for step_idx, step in enumerate(gov.history, start=1):
                writer.writerow([
                    step_idx,
                    repr(step["observation"]),
                    repr(step["raw_adjustments"]),
                    repr(step["adjustments"]),
                    f"{step['reward']:.6f}",
                    step["death_count"],
                ])
        print(f"Saved governor history CSV to: {file_path}")

    def print_summary(self):
        print("==== Experiment Summary ====")
        for i, total in enumerate(self.total_rewards):
            print(f"Agent {i} ({self.agents[i].name}) total reward: {total:.2f}")
        print("--------------")
        
        print("Agent wealths and alive status:")
        for i, (wealth, alive) in enumerate(zip(self.env.agent_wealths, self.env.is_alive)):
            status = "alive" if alive else "dead"
            print(f"  Agent {i} - wealth: {wealth:.2f} - {status}")
        print("----------------------------")
        
        total_combined_wealth = sum(self.env.agent_wealths)
        print(f"Total Combined Wealth (All Agents): {total_combined_wealth:.2f}")
        print("----------------------------")
        
        total_combined_reward = sum(self.total_rewards)
        print(f"Total Combined Reward (All Agents): {total_combined_reward:.2f}")

        # Governor debug summary display
        gov = self.env.governor
        if gov and hasattr(gov, "history") and gov.history:
            print("----------------------------")
            print("Governor debug summary:")
            if hasattr(gov, "means"):
                print(f"  Policy means: {[f'{m:.4f}' for m in gov.means]}")
            print(f"  Total governor steps recorded: {len(gov.history)}")

            for step_idx, step in enumerate(gov.history[:5], start=1):
                print(f"  Step {step_idx}: raw={step['raw_adjustments']}, adjusted={step['adjustments']}, reward={step['reward']:.2f}, deaths={step['death_count']}")

            avg_adjustments = [sum(step['adjustments'][i] for step in gov.history) / len(gov.history)
                               for i in range(len(self.agents))]
            print(f"  Avg governor adjustments: {[f'{a:.4f}' for a in avg_adjustments]}")

        print("----------------------------")
        print("Agent survival duration:")
        for i, death_step in enumerate(self.env.death_steps):
            if death_step is None:
                print(f"  Agent {i} survived all {self.T} steps")
            else:
                print(f"  Agent {i} died at step {death_step}")
        print("============================")
        #print(dir(self.agents))

    def plot_reward_trajectories(self):
        import numpy as np
        plt.figure(figsize=(12, 6))
        
        # Use a colormap to smoothly distribute colors across any number of agents
        cmap = plt.cm.get_cmap('tab20', len(self.agents))
        
        for agent_idx, agent in enumerate(self.agents):
            rewards = np.array([step[agent_idx] for step in self.rewards_log])
            cumavg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
            
            # If agent died early, truncate the line at their death step to keep data clean
            death_step = self.env.death_steps[agent_idx]
            if death_step is not None:
                plt.plot(cumavg[:death_step], color=cmap(agent_idx), alpha=0.8)
                plt.scatter(death_step - 1, cumavg[death_step - 1], color='red', marker='x', s=20, zorder=3)
            else:
                plt.plot(cumavg, label=f"Ag {agent_idx}" if len(self.agents) <= 10 else "", color=cmap(agent_idx), alpha=0.6)
                
        plt.title("Cumulative Average Reward Per Agent Over Time", fontsize=12, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Reward")
        if len(self.agents) <= 10:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "reward_trajectories.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_arm_frequencies(self):
        import numpy as np
        import seaborn as sns
        n_agents = len(self.agents)
        n_arms = self.env.n_arms
        
        counts = np.zeros((n_agents, n_arms), dtype=int)
        for step_idx, step in enumerate(self.choices_log):
            for agent_idx, arm in enumerate(step):
                # FIX: Ignore ghost pulls if the arm data is None or negative (dead state placeholder)
                if arm is None or arm < 0:
                    continue
                counts[agent_idx, arm] += 1

        # FIX: For high scale, subplots crush text. Transform into a consolidated Heatmap instead!
        if n_agents > 5:
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(counts, cmap="YlGnBu", xticklabels=5, yticklabels=5, cbar_kws={'label': 'Pull Count'})
            plt.title("Arm Pull Frequency Matrix (Agents vs Arms)", fontsize=12, fontweight='bold')
            plt.xlabel("Arm Index")
            plt.ylabel("Agent Index")
            plt.tight_layout()
        else:
            fig, axes = plt.subplots(n_agents, 1, figsize=(10, 2 * n_agents), sharex=True)
            if n_agents == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.bar(range(n_arms), counts[i], color=f"C{i % 10}")
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title(f"Arm frequencies — {self.agents[i].name} (Agent {i})", fontsize=10)
                ax.grid(True, axis='y', linestyle=":", alpha=0.5)
            axes[-1].set_xlabel("Arm Index")
            plt.tight_layout()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "arm_frequencies.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_agent_beliefs(self):
        """Group agents by survival profiles and plot a macro-economic map of their internal beliefs."""
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.values_log:
            print("No internal agent values logged to plot.")
            return

        n_agents = len(self.agents)
        n_arms = self.env.n_arms
        
        # 1. Extract the final recorded beliefs for every agent.
        # If an agent died at step X, we pull their beliefs from step X-1.
        # If they survived, we pull from the final step (T-1).
        final_beliefs = np.zeros((n_agents, n_arms))
        
        for agent_idx in range(n_agents):
            death_step = self.env.death_steps[agent_idx]
            if death_step is not None:
                # Agent died: extract their final state of mind right before death
                final_beliefs[agent_idx, :] = self.values_log[death_step - 1][agent_idx]
            else:
                # Agent survived: extract their final end-of-game beliefs
                final_beliefs[agent_idx, :] = self.values_log[-1][agent_idx]

        # 2. Categorize agents into 3 distinct survival classes
        elite_indices = []       # Survived all 1000 steps
        struggling_indices = []   # Survived a moderate duration (> 150 steps but died)
        starved_indices = []      # Died early (<= 150 steps)

        for idx, death_step in enumerate(self.env.death_steps):
            if death_step is None:
                elite_indices.append(idx)
            elif death_step > 150:
                struggling_indices.append(idx)
            else:
                starved_indices.append(idx)

        # 3. Compute class-wide average beliefs across all 30 arms
        class_matrix = np.zeros((3, n_arms))
        class_labels = [
            f"Elite Survivors (N={len(elite_indices)})",
            f"Struggling Mid-Class (N={len(struggling_indices)})",
            f"Early Starved Class (N={len(starved_indices)})"
        ]

        class_matrix[0, :] = np.mean(final_beliefs[elite_indices, :], axis=0) if elite_indices else 0
        class_matrix[1, :] = np.mean(final_beliefs[struggling_indices, :], axis=0) if struggling_indices else 0
        class_matrix[2, :] = np.mean(final_beliefs[starved_indices, :], axis=0) if starved_indices else 0

        # 4. Generate the Clustered Macro Belief Heatmap
        plt.figure(figsize=(14, 5))
        
        # Using a bright colormap to represent perceived valuation (e.g., "viridis" or "magma")
        sns.heatmap(
            class_matrix, 
            cmap="viridis", 
            xticklabels=2,
            yticklabels=class_labels,
            annot=False, 
            cbar_kws={'label': 'Average Perceived Reward Value'}
        )

        plt.title("Socio-Economic Belief Mapping: Final Perceived Arm Values by Agent Class", fontsize=12, fontweight='bold')
        plt.xlabel("Arm Index (Sorted Best to Worst Actual Base Mean)")
        plt.ylabel("Agent Survival Class")
        
        # Rotate y-axis labels for readability
        plt.yticks(rotation=0)
        plt.tight_layout()

        if self.save_dir:
            save_path = os.path.join(self.save_dir, "agent_beliefs_macro_classes.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Visualizer] Upgraded agent beliefs plot saved to: {save_path}")

        plt.show()

    def plot_environmental_health(self):
        """Plots a macro-tier heatmap and summary bands of environmental health over time.
        
        Transforms a 30-line chaotic line plot into a readable macro-economic digest.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not hasattr(self.env, "environmental_health_history") or not self.env.environmental_health_history:
            print("[Visualizer] Connected environment does not contain environmental health logs.")
            return

        # Shape: (Timesteps, 30)
        health_history = np.array(self.env.environmental_health_history)
        timesteps = len(health_history)
        n_arms = self.env.n_arms

        # 1. Segment arms into three clean, logical resource tiers
        # Tier 1: Premium (0-9), Tier 2: Mid-Grade (10-19), Tier 3: Subsistence (20-29)
        tier_size = max(1, n_arms // 3)
        t1_health = np.mean(health_history[:, :tier_size], axis=1)
        t2_health = np.mean(health_history[:, tier_size:2*tier_size], axis=1)
        t3_health = np.mean(health_history[:, 2*tier_size:], axis=1)

        # Create a double-paneled figure (Top: Heatmap of all arms, Bottom: Tier Averages)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.2, 1]})

        # --- PANEL 1: Macro Heatmap of Ecosystem Degradation ---
        # Downsample timesteps if T is huge to make the heatmap render smoothly
        step_interval = max(1, timesteps // 100) 
        heatmap_data = health_history[::step_interval, :].T  # Shape: (Arms, Sampled Steps)
        
        sns.heatmap(
            heatmap_data, 
            cmap="RdYlGn", # Green = Healthy (1.0), Red = Fully Depleted (0.0)
            vmin=0.0, vmax=1.0,
            ax=ax1, 
            cbar_kws={'label': 'Resource Capacity Factor (κ)'}
        )
        
        ax1.set_title("Ecosystem Breakdown: Resource Capacity Matrix (All 30 Arms)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Arm Index (Sorted Best to Worst)")
        # Fix X-axis labels to show true simulation steps rather than array indices
        x_ticks = np.arange(0, heatmap_data.shape[1], max(1, heatmap_data.shape[1] // 10))
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([str(i * step_interval) for i in x_ticks])

        # --- PANEL 2: Consolidated Tier Trends ---
        steps = np.arange(timesteps) + 1
        ax2.plot(steps, t1_health, label="Tier 1: Premium Assets (Arms 0-9)", color='#2b8cbe', linewidth=2)
        ax2.plot(steps, t2_health, label="Tier 2: Mid-Tier Assets (Arms 10-19)", color='#fe9929', linewidth=2)
        ax2.plot(steps, t3_health, label="Tier 3: Subsistence Assets (Arms 20-29)", color='#41ab5d', linewidth=2)
        
        # Draw a baseline reference line at the threshold where system strain might happen
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

        ax2.set_title("Macro-Tier Resource Health Dynamics", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Simulation Timestep")
        ax2.set_ylabel("Avg Capacity Factor (κ)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower left", fontsize=9)

        plt.tight_layout()

        if self.save_dir:
            save_path = os.path.join(self.save_dir, "environmental_health_upgraded.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Visualizer] Upgraded environmental health plot saved to: {save_path}")

        plt.show()

    def plot_resource_efficiency(self):
        """Plots a 3-layer macro stacked area chart showing Extracted Wealth, 
        Environmental Degradation, and Opportunity Costs against a flat static ceiling,
        complete with an overlaid population survival line. Saves output to save_dir.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        if not hasattr(self.env, "environmental_health_history") or not self.env.environmental_health_history:
            print("[Visualizer] Missing environmental health history.")
            return

        health_history = np.array(self.env.environmental_health_history) # (T, n_arms)
        timesteps = len(health_history)
        n_arms = self.env.n_arms
        n_agents = len(self.agents)

        # 1. Extract the TRUE baseline means directly from your configured environment
        if hasattr(self.env, "arms"):
            base_means = np.array([arm.mean for arm in self.env.arms])
        else:
            # Fallback hardcoded to match your precise terminal printout exactly
            base_means = np.array([
                10.0, 9.0, 8.2, 7.5, 6.8, 5.5, 5.0, 4.6, 4.2, 3.9,
                3.6,  3.3, 3.0, 2.8, 2.6, 2.0, 1.9, 1.8, 1.7, 1.6,
                1.5,  1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5
            ])
        
        # 2. True Global Optimal calculated from your true environment total (99.4)
        global_optimal_constant = np.sum(base_means)
        theoretical_ceiling = np.full(timesteps, global_optimal_constant)

        # Track agent survival states
        alive_history = np.zeros((timesteps, n_agents), dtype=bool)
        for i, death_step in enumerate(self.env.death_steps):
            if death_step is None:
                alive_history[:, i] = True
            else:
                alive_history[:death_step, i] = True

        # 3. Initialize our three exclusive structural layers
        net_extracted_wealth = np.zeros(timesteps)
        environmental_degradation_loss = np.zeros(timesteps)
        opportunity_cost_loss = np.zeros(timesteps)

        for t in range(timesteps):
            # If all agents are dead, the entire environment baseline is lost to opportunity cost
            if np.sum(alive_history[t, :]) == 0:
                opportunity_cost_loss[t] = global_optimal_constant
                continue

            # Extract actual rewards taken home by agents at step t
            if hasattr(self, "rewards_log") and len(self.rewards_log) > t:
                step_rewards = np.array(self.rewards_log[t])
                living_rewards = step_rewards[alive_history[t, :]]
                net_extracted_wealth[t] = np.sum(living_rewards)

            # Look at what arms were actually chosen by the living agents at this step
            if hasattr(self, "choices_log") and len(self.choices_log) > t:
                step_choices = np.array(self.choices_log[t])
                living_choices = step_choices[alive_history[t, :]]
                
                # Force conversion to integer array for accurate indexing slices
                unique_chosen_arms = np.unique(living_choices).astype(int)
                unchosen_arms = np.setdiff1d(np.arange(n_arms), unique_chosen_arms).astype(int)
                
                # Layer A: Opportunity Cost = Baseline values of completely abandoned arms
                opportunity_cost_loss[t] = np.sum(base_means[unchosen_arms])
                
                # Layer B: Environmental Degradation = The capacity drop on the chosen arms
                kappa_t = health_history[t, :]
                chosen_arms_capacity_loss = base_means[unique_chosen_arms] * (1.0 - kappa_t[unique_chosen_arms])
                environmental_degradation_loss[t] = np.sum(chosen_arms_capacity_loss)
                
                # Micro-adjustment to prevent rounding/sampling overlaps from breaking the stack boundaries
                calculated_total = net_extracted_wealth[t] + opportunity_cost_loss[t] + environmental_degradation_loss[t]
                if calculated_total > global_optimal_constant:
                    opportunity_cost_loss[t] = max(0, global_optimal_constant - net_extracted_wealth[t] - environmental_degradation_loss[t])

        # 4. Plot the 3-Layer Stacked Graph
        fig, ax1 = plt.subplots(figsize=(13, 7))
        steps = np.arange(timesteps) + 1

        # Plot Stacked Areas
        ax1.stackplot(
            steps, 
            net_extracted_wealth, 
            environmental_degradation_loss, 
            opportunity_cost_loss,
            labels=[
                'Net System Wealth Extracted (Blue)', 
                'Value Lost to Environmental Degradation (Orange)', 
                'Value Lost to Leftover Opportunity Cost (Red)'
            ],
            colors=['#2b8cbe', '#fe9929', '#e34a33'], 
            alpha=0.85
        )

        # Plot the true constant ceiling line
        ax1.plot(steps, theoretical_ceiling, label='Theoretical Pareto-Optimal Capacity Ceiling', color='black', linestyle='--', linewidth=2.5)

        # Axis styling
        ax1.set_title("Macroscopic Resource Utilization & Decentralized Waste Breakdown", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Simulation Timestep", fontsize=11)
        ax1.set_ylabel("Total Resource Value (Units of Reward)", fontsize=11)
        ax1.set_xlim(0, timesteps)
        ax1.set_ylim(0, global_optimal_constant * 1.05)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # 5. Create a twin axis to overlay population dynamics smoothly
        ax2 = ax1.twinx()
        ax2.plot(steps, np.sum(alive_history, axis=1), color='#555555', linestyle=':', linewidth=2, label='Active Living Population')
        ax2.set_ylabel('Active Population Count', color='#555555', fontsize=11)
        ax2.set_ylim(0, n_agents * 1.05)

        # Consolidate legends from both independent y-axes into a single display box
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
        
        plt.tight_layout()

        # 6. Save block implementation
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "macro_resource_efficiency.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Visualizer] Upgraded resource efficiency plot saved to: {save_path}")

        plt.show()