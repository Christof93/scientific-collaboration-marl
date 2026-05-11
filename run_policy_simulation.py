"""
Example script showing how to use the agent policies with the peer group environment.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from agent_policies import (create_mixed_policy_population,
                            create_per_group_policy_population,
                            do_nothing_policy, get_policy_function)
from env.peer_group_environment import PeerGroupEnvironment
from log_simulation import SimLog
from stats_tracker import SimulationStats

# Define different policy distributions to test
POLICY_CONFIGS = {
    "All Careerist": {
        "careerist": 1.0,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.0,
    },
    "All Orthodox": {
        "careerist": 0.0,
        "orthodox_scientist": 1.0,
        "mass_producer": 0.0,
    },
    "All Mass Producer": {
        "careerist": 0.0,
        "orthodox_scientist": 0.0,
        "mass_producer": 1.0,
    },
    "Balanced": {
        "careerist": 1 / 3,
        "orthodox_scientist": 1 / 3,
        "mass_producer": 1 / 3,
    },
    "Careerist Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.5,
        "mass_producer": 0.0,
    },
    "Orthodox Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.5,
    },
    "Mass Producer Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.5,
    },
}


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

    # Create environment
    env = PeerGroupEnvironment(
        start_agents=start_agents,
        max_steps=max_steps,
        max_agents=n_agents,
        n_groups=n_groups,
        max_peer_group_size=max_peer_group_size,
        n_projects_per_step=1,
        max_projects_per_agent=8,
        max_agent_age=750,
        max_rewardless_steps=max_rewardless_steps,
        acceptance_threshold=acceptance_threshold,
        coordination_factor=coordination_factor,
        continuation_probability=continuation_probability,
        reward_type=reward_type,
        distribution_mode=distribution_mode,
    )
    if group_policy_homogenous:
        # Create agent policy assignments
        agent_policies = create_per_group_policy_population(
            n_agents, policy_distribution
        )
    else:
        agent_policies = create_mixed_policy_population(
            n_agents, policy_distribution, seed=seed
        )
    if verbose:
        print(
            f"Agent policy distribution: {dict(zip(*np.unique(agent_policies, return_counts=True)))}"
        )

    # Initialize stats tracker
    stats = SimulationStats()

    log = SimLog(
        "log",
        f"{output_file_prefix}_actions.jsonl",
        f"{output_file_prefix}_observations.jsonl",
        f"{output_file_prefix}_projects.json",
    )
    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration")):
        log.start()

    # Reset environment
    observations, infos = env.reset(seed=seed)

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
        if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration")):
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

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration")):
        env.area.save(f"log/{output_file_prefix}_area.pickle")

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration")):
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

    if not (output_file_prefix.startswith("sensitivity") or output_file_prefix.startswith("calibration")):
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


def compare_policy_performances():
    """Compare the performance of different policy distributions."""

    results = {}

    for config_name, policy_dist in POLICY_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {config_name}")
        print(f"{'='*50}")

        result = run_simulation_with_policies(
            n_agents=2_000,
            start_agents=100,
            max_steps=5_000,
            n_groups=50,
            max_peer_group_size=100,
            policy_distribution=policy_dist,
            output_file_prefix=f"policy_{config_name.lower().replace(' ', '_')}",
        )

        results[config_name] = result["final_stats"]

    # Print comparison
    print(f"\n{'='*80}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*80}")

    for config_name, stats in results.items():
        success_rate = stats["successful_projects"] / max(stats["finished_projects"], 1)
        print(
            f"{config_name:20} | Success Rate: {success_rate:.3f} | "
            f"Finished: {stats['finished_projects']:3d} | "
            f"Rewards: {stats['total_rewards_distributed']:6.2f}"
        )


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
        output_file_prefix=f"balanced_{reward_type}_{distribution_mode}_seed{seed}",
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

CALIBRATED_PARAMS={
    "reputation": [('acceptance_threshold', 0.9070562642422293), ('orthodox_novelty_threshold', 0.4661882540054239), ('careerist_prestige_threshold', 0.5288947013824785), ('mass_producer_effort_threshold', np.int64(16)), ('max_rewardless_steps', np.int64(53)), ('coordination_factor', 0.18377486003694873), ('continuation_probability', 0.4036963525119355)],
    "raw_pubcount": [('acceptance_threshold', 0.7625441136962994), ('orthodox_novelty_threshold', 0.4), ('careerist_prestige_threshold', 0.5563978708007807), ('mass_producer_effort_threshold', np.int64(12)), ('max_rewardless_steps', np.int64(50)), ('coordination_factor', 0.58953459329011), ('continuation_probability', 0.21996776415961516)],
    "h_index": [('acceptance_threshold', 0.7001955006534182), ('orthodox_novelty_threshold', 0.7569754628832838), ('careerist_prestige_threshold', 0.4949222510564053), ('mass_producer_effort_threshold', np.int64(21)), ('max_rewardless_steps', np.int64(64)), ('coordination_factor', 0.10039137685896896), ('continuation_probability', 0.2460463010038782)],
    "all": [('acceptance_threshold', 1.1637780469746222), ('orthodox_novelty_threshold', 0.4), ('careerist_prestige_threshold', 0.6), ('mass_producer_effort_threshold', np.int64(20)), ('max_rewardless_steps', np.int64(50)), ('coordination_factor', 0.1), ('continuation_probability', 0.5190629152756893)]
}
REWARD_TYPE = "all"
DISTRIBUTION_MODE = "multiply"

if __name__ == "__main__":
    
    cp = {k:v for k,v in CALIBRATED_PARAMS[REWARD_TYPE]}
    # Choose between running a single simulation or the full batch
    run_simulation_with_policies(
        n_agents=3000,
        start_agents=200,
        max_steps=600,
        n_groups=20,
        max_peer_group_size=150,
        policy_distribution={
            "random": 0.5,
            "careerist": 0.5
        },
        output_file_prefix=f"balanced_{REWARD_TYPE}_{DISTRIBUTION_MODE}_careerist_vs_random_seed42",
        group_policy_homogenous=True,
        reward_type=REWARD_TYPE,
        distribution_mode=DISTRIBUTION_MODE,
        seed=42,
        max_rewardless_steps=cp["max_rewardless_steps"],
        acceptance_threshold=cp["acceptance_threshold"],
        novelty_threshold=cp["orthodox_novelty_threshold"],
        prestige_threshold=cp["careerist_prestige_threshold"],
        effort_threshold=cp["mass_producer_effort_threshold"],
        coordination_factor=cp["coordination_factor"],
        continuation_probability=cp["continuation_probability"],
    )

    # Run simulation for all reward functions on random seeds in parallel
    # run_all_reward_functions(cp, r_type = REWARD_TYPE, seeds=range(10), n_workers=10, distribution_modes=["multiply"])
    # run_all_reward_functions(cp, r_type = "reputation", seeds=range(10), n_workers=30, distribution_modes=["multiply"])
    # run_all_reward_functions(cp, r_type = "raw_pubcount", seeds=range(10), n_workers=30, distribution_modes=["multiply"])
    # run_all_reward_functions(cp, r_type = "h_index", seeds=range(10), n_workers=30, distribution_modes=["multiply"])
