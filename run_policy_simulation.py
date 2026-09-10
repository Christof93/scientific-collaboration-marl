"""
script to run the peer group environment experiments.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from agent_policies import (create_mixed_policy_population,
                            create_per_group_policy_population,
                            do_nothing_policy, get_policy_function,
                            validate_proportions)
from env.peer_group_environment import PeerGroupEnvironment
from log_simulation import SimLog
from stats_tracker import SimulationStats

def run_simulation_with_policies(
    n_agents: int = 100,
    max_steps: int = 1_000,
    max_rewardless_steps: int = 250,
    start_agents: int = 60,
    n_groups: int = 8,
    max_peer_group_size: int = 40,
    policy_distribution: dict = None,
    output_file_prefix: str = None,
    group_policy_homogenous=True,
    acceptance_threshold: float = 0.5,
    novelty_threshold: float = 0.8,
    prestige_threshold: float = 0.2,
    effort_threshold: int = 22,
    seed:int=42,
    reward_type: str = "all",
    distribution_mode: str = "multiply",
    coordination_factor: float = 0.2,
    continuation_probability: float = 0.5,
    ratio_group_expansion_depends_on_success: float = 0.5,
    prestige_eval_noise_factor: float = 0.1,
    verbose: bool = True,
):
    """
    Run a simulation with different agent policies.

    Args:
        n_agents: Number of agents in the simulation
        max_steps: Maximum number of simulation steps
        policy_distribution: Distribution of policies among agents
        output_file: File to save results
    """

    if group_policy_homogenous:
        # Create agent policy assignments
        agent_to_group, agent_policies = create_per_group_policy_population(
            n_agents, n_groups, policy_distribution, seed=seed
        )
    else:
        agent_to_group, agent_policies = create_mixed_policy_population(
            n_agents, n_groups, policy_distribution, seed=seed
        )
    if verbose:
        print(
            f"Agent policy distribution: {dict(zip(*np.unique(agent_policies, return_counts=True)))}"
        )
        validate_proportions(agent_to_group, agent_policies)

    # Create environment
    env = PeerGroupEnvironment(
        start_agents=start_agents,
        max_steps=max_steps,
        max_agents=n_agents,
        n_groups=n_groups,
        agent_to_group=agent_to_group,
        max_peer_group_size=np.unique(agent_to_group, return_counts=True)[1].max(),
        n_projects_per_step=1,
        max_projects_per_agent=8,
        max_agent_age=750,
        max_rewardless_steps=max_rewardless_steps,
        acceptance_threshold=acceptance_threshold,
        coordination_factor=coordination_factor,
        continuation_probability=continuation_probability,
        reward_type=reward_type,
        distribution_mode=distribution_mode,
        ratio_group_expansion_depends_on_success=ratio_group_expansion_depends_on_success,
        prestige_eval_noise_factor=prestige_eval_noise_factor,
    )
    # Initialize stats tracker
    stats = SimulationStats()

    log = SimLog(
        "log",
        f"{output_file_prefix}_actions.jsonl",
        f"{output_file_prefix}_observations.jsonl",
        f"{output_file_prefix}_projects.json",
    )
    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration") or output_file_prefix.startswith("tipping_point")):
        log.start()

    # Reset environment
    observations, infos = env.reset(seed=seed)
    if verbose:
        print(env.max_peer_group_size)

    # Simulation loop
    for step in range(max_steps):
        actions = {}

        # Generate actions for each agent based on their policy
        for agent in env.agents:
            agent_idx = env.agent_to_id[agent]
            policy_name = agent_policies[agent_idx]
            if env.active_agents[agent_idx] == 0:
                policy_func = do_nothing_policy
                policy_name = None
            else:
                policy_func = get_policy_function(policy_name)

            # Get agent's observation and action mask
            obs = observations[agent]["observation"]
            action_mask = observations[agent]["action_mask"]
            # Generate action using the agent's policy
            if policy_name == "careerist":
                action = policy_func(obs, action_mask, prestige_threshold)
            elif policy_name == "orthodox_scientist":
                action = policy_func(obs, action_mask, novelty_threshold)
            elif policy_name == "mass_producer":
                action = policy_func(obs, action_mask, effort_threshold)
            else:
                action = policy_func(obs, action_mask)

            actions[agent] = action

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # if step > 500:
        #     active_agent_1 = list(env.active_agents).index(1)
        #     print(env.action_masks[f"agent_{active_agent_1}"])
        if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration") or output_file_prefix.startswith("tipping_point")):
            log.log_observation(
                {
                    a: obs if env.active_agents[env.agent_to_id[a]] == 1 else None
                    for a, obs in observations.items()
                }
            )
            log.log_action(
                {
                    a: (
                        act | {"archetype": agent_policies[env.agent_to_id[a]]}
                        if env.active_agents[env.agent_to_id[a]] == 1
                        else None
                    )
                    for a, act in actions.items()
                }
            )
        # Update stats
        stats.update(env, observations, rewards, terminations, truncations)

        # Print progress
        if step % 100 == 0 and verbose:
            print(f"Step {step}: {stats.summary_line()}")

        # Check if all agents are done
        if all(terminations.values()):
            if verbose:
                print(f"Simulation ended at step {step}")
            break

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration") or output_file_prefix.startswith("tipping_point")):
        env.area.save(f"log/{output_file_prefix}_area.pickle")

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration") or output_file_prefix.startswith("tipping_point")):
        log.log_projects(env.projects.values())
    # Calculate active agent populations
    active_mask = env.active_agents.astype(bool)
    active_policies = [agent_policies[i] for i, active in enumerate(active_mask) if active]
    unique_pols, counts = np.unique(active_policies, return_counts=True)
    active_populations = dict(zip(unique_pols, counts.tolist()))
    print(active_populations)
    # Save results
    results = {
        "final_stats": stats.to_dict(),
        "agent_policies": agent_policies,
        "policy_populations": active_populations,
        "policy_distribution": policy_distribution
        or {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    }

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration") or output_file_prefix.startswith("tipping_point")):
        with open("log/" + output_file_prefix + "_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    if verbose:
        print(f"\nFinal Results:")
        print(f"Total Steps: {stats.total_steps}")
        print(f"Finished Projects: {stats.finished_projects_count}")
        print(f"Successful Projects: {stats.successful_projects_count}")
        print(
            f"Success Rate: {stats.successful_projects_count / max(stats.finished_projects_count, 1):.3f}"
        )
        print(f"Total Rewards: {stats.total_rewards_distributed:.2f}")
    results["projects"] = [p.to_dict() for p in env.projects.values()]
    return results


def run_simulation_worker(args):
    """Worker function for parallel simulation runs."""
    params, seed, reward_type, distribution_mode = args
    print(f"--- Starting: {reward_type}/{distribution_mode} (seed {seed}) ---")
    run_simulation_with_policies(
        n_agents=3000,
        start_agents=200,
        max_steps=600,
        n_groups=20,
        max_peer_group_size=150,
        policy_distribution=params["policy_distribution"] if "policy_distribution" in params else {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        },
        output_file_prefix=f"{params['log_prefix'] if 'log_prefix' in params else 'balanced'}_{reward_type}_{distribution_mode}_seed{seed}",
        group_policy_homogenous=False,
        reward_type=reward_type,
        distribution_mode=distribution_mode,
        seed=seed,
        max_rewardless_steps=params["max_rewardless_steps"],
        acceptance_threshold=params["acceptance_threshold"],
        novelty_threshold=params["orthodox_novelty_threshold"],
        prestige_threshold=params["careerist_prestige_threshold"],
        effort_threshold=params["mass_producer_effort_threshold"],
        coordination_factor=params["coordination_factor"],
        continuation_probability=params["continuation_probability"],
        verbose=False,  # Set to False for parallel execution
    )
    print(f"--- Finished: {reward_type}/{distribution_mode} (seed {seed}) ---")


def run_all_reward_functions(parameters, r_type, seeds=range(10), n_workers=8, distribution_modes = [
        "multiply",
        "evenly",
        "by_effort"
    ]):
    """Run simulations for all combinations of reward types and distribution modes in parallel."""
    

    tasks = []
    for seed in seeds:
        for d_mode in distribution_modes:
            tasks.append((parameters, seed, r_type, d_mode))

    print(f"Starting parallel execution of {len(tasks)} simulations with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(run_simulation_worker, tasks))

    print("All simulations completed.")

def build_reward_summary_by_archetype(reward_steps, agents, seed, strategy):
    """
    Returns a DataFrame with rows:
      step, archetype, mean_reward, std_reward, n_agents, seed, strategy

    Raises:
        ValueError if any agent has no archetype mapping.
    """
    # === 1. Build agent_id -> archetype map ===
    agent_archetype = {}
    for a in agents:
        if not isinstance(a, dict):
            continue
        for agent_id, v in a.items():
            if v is None:
                continue
            if isinstance(v, dict) and "archetype" in v:
                agent_archetype[agent_id] = v["archetype"]

    if not agent_archetype:
        raise ValueError("No archetypes found in agents data.")

    # === 2. Build per-step stats grouped by archetype ===
    records = []
    for step_idx, step in enumerate(reward_steps):
        if not isinstance(step, dict):
            continue

        # Gather rewards per archetype for this step
        arch_rewards = {}
        for agent_id, data in step.items():
            if data is None:
                continue

            # Every agent MUST have an archetype
            if agent_id not in agent_archetype:
                raise ValueError(f"Missing archetype for agent_id '{agent_id}' at step {step_idx}")

            obs = data.get("observation", {}) if isinstance(data, dict) else {}
            if "accumulated_rewards" not in obs:
                continue

            reward = obs["accumulated_rewards"][0]
            archetype = agent_archetype[agent_id]
            arch_rewards.setdefault(archetype, []).append(reward)

        # Compute mean/std per archetype
        for archetype, rewards in arch_rewards.items():
            s = pd.Series(rewards)
            records.append({
                "step": step_idx,
                "archetype": archetype,
                "mean_reward": float(s.mean()),
                "std_reward": float(s.std(ddof=1)),  # sample std
                "n_agents": len(rewards),
                "seed": int(seed),
                "strategy": strategy,
            })

    return pd.DataFrame(records)

def build_reward_dataframe(reward_steps, agents, seed):
    """
    Builds a DataFrame of accumulated rewards per agent per step,
    annotated with archetype.
    """
    agent_archetype = {}
    for a in agents:
        for k, v in a.items():
            if v is not None:
                agent_archetype[k] = v["archetype"]

    records = []
    for step_idx, step in enumerate(reward_steps):
        for agent_id, data in step.items():
            if data is not None:
                data = data.get("observation", None)
                if data and "accumulated_rewards" in data:
                    archetype = agent_archetype.get(agent_id, None)
                    if archetype is not None:
                        records.append({
                            "step": step_idx,
                            "archetype": archetype,
                            "agent_id": agent_id,
                            "accumulated_rewards": data["accumulated_rewards"][0],
                            "h_index": data["peer_h_index"][0],
                            "age": data["age"][0],
                            # "accumulated_citations": len(data.get("citations", [])),
                            # "societal_value": data['societal_value_score'],
                            "seed": seed,
                        })

    return pd.DataFrame(records)

def save_results():
    ## save trajectories
    dfs_all = {}
    write_files = True
    name = "multiply"
    for reward_type in [
        "all",
        "raw_pubcount", 
        "reputation", 
        "h_index"
    ]:
        all_summaries = []
        all_rewards = []
        for seed in range(30):
            try:
                with open(f"log/balanced_{reward_type}_{name}_seed{seed}_actions.jsonl", "r") as f:
                    balanced_actions = [json.loads(line) for line in f]
                with open(f"log/balanced_{reward_type}_{name}_seed{seed}_observations.jsonl", "r") as f:
                    balanced_observations = [json.loads(line) for line in f]
                df_all = build_reward_dataframe(balanced_observations, balanced_actions, seed)
                all_rewards.append(df_all)
            except FileNotFoundError:
                print("log files of 30 seed runs could not be located!")
                write_files = False
                break
        df_all = pd.concat(all_rewards, ignore_index=True)
        dfs_all[reward_type] = df_all
        if write_files:
            df_all.to_parquet(f"results/reward_trajectories_{reward_type}_{name}.parquet", index=False)
            print(f"Saved {name} simulation to reward_trajectories_{reward_type}_{name}.parquet "
                    f"({len(df_all)} records).")
    ## save summaries
    df_summary_all = {}
    dfs_summary = {}
    write_files = True
    name = "multiply"
    for reward_type in [
        "raw_pubcount", 
        "reputation", 
        "h_index",
        "all"
    ]:
        all_summaries = []
        all_rewards = []
        for seed in range(30):
            try:
                with open(f"log/balanced_{reward_type}_{name}_seed{seed}_actions.jsonl", "r") as f:
                    balanced_actions = [json.loads(line) for line in f]
                with open(f"log/balanced_{reward_type}_{name}_seed{seed}_observations.jsonl", "r") as f:
                    balanced_observations = [json.loads(line) for line in f]

                df_summary = build_reward_summary_by_archetype(
                    balanced_observations, balanced_actions, seed, reward_type
                )
                all_summaries.append(df_summary)
            except FileNotFoundError:
                print("log files of 30 seed runs could not be located!")
                write_files = False
                break
        df_summary_all = pd.concat(all_summaries, ignore_index=True)
        dfs_summary[reward_type] = df_summary_all
        if write_files:
            df_summary_all.to_parquet(f"results/reward_summary_by_archetype_{reward_type}_{name}.parquet", index=False)
            print(f"Saved {name} summary to reward_summary_by_archetype_{reward_type}_{name}.parquet "
                f"({len(df_summary_all)} records across {len(all_summaries)} seeds).")

CALIBRATED_PARAMS={
    "all":[('acceptance_threshold', 1.2110517170409714), ('orthodox_novelty_threshold', 0.15), ('careerist_prestige_threshold', 0.6), ('mass_producer_effort_threshold', np.int64(17)), ('max_rewardless_steps', np.int64(84)), ('coordination_factor', 0.1), ('continuation_probability', 0.28098555154267013)]
}
REWARD_TYPE = "all"
DISTRIBUTION_MODE = "multiply"

if __name__ == "__main__":
    
    cp = {k:v for k,v in CALIBRATED_PARAMS[REWARD_TYPE]}
    # Choose between running a single simulation or the full batch
    # run_simulation_with_policies(
    #     n_agents=3000,
    #     start_agents=200,
    #     max_steps=600,
    #     n_groups=20,
    #     max_peer_group_size=150,
    #     policy_distribution={
    #         "random": 0.5,
    #         "careerist": 0.5,
    #         # "orthodox_scientist": 0.25,
    #         # "mass_producer": 0.25,
    #     },
    #     output_file_prefix=f"balanced_{REWARD_TYPE}_{DISTRIBUTION_MODE}_careerist_vs_random_seed42",
    #     group_policy_homogenous=True,
    #     reward_type=REWARD_TYPE,
    #     distribution_mode=DISTRIBUTION_MODE,
    #     seed=0,
    #     max_rewardless_steps=cp["max_rewardless_steps"],
    #     acceptance_threshold=cp["acceptance_threshold"],
    #     novelty_threshold=cp["orthodox_novelty_threshold"],
    #     prestige_threshold=cp["careerist_prestige_threshold"],
    #     effort_threshold=cp["mass_producer_effort_threshold"],
    #     coordination_factor=cp["coordination_factor"],
    #     continuation_probability=cp["continuation_probability"],
    # )
    
    # Run simulation for all reward functions on random seeds in parallel
    run_all_reward_functions(cp, r_type = "reputation", seeds=range(30), n_workers=30, distribution_modes=["multiply"])
    run_all_reward_functions(cp, r_type = "raw_pubcount", seeds=range(30), n_workers=30, distribution_modes=["multiply"])
    run_all_reward_functions(cp, r_type = "h_index", seeds=range(30), n_workers=30, distribution_modes=["multiply"])
    run_all_reward_functions(cp, r_type = "all", seeds=range(30), n_workers=30, distribution_modes=["multiply"])

    # careerist vs random
    cp["policy_distribution"] = {
            "random": 0.5,
            "careerist": 0.5,
    }
    cp["log_prefix"] = "careerist_vs_random"
    run_all_reward_functions(cp, r_type = REWARD_TYPE, seeds=range(30), n_workers=30, distribution_modes=["multiply"])
    save_results()
    ### ppython run_policy_simulation.py  135827.16s user 1210.73s system 751% cpu 5:04:04.14 total