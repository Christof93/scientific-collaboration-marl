import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from run_policy_simulation import run_simulation_with_policies, CALIBRATED_PARAMS


def get_config():
    dry_run = os.environ.get("TIPPING_POINT_DRY_RUN", "true").lower() in {"1", "true", "yes", "on"}
    if dry_run:
        return {
            "seeds": [0],
            "proportions": [0.15, 0.95],
        }
    return {
        "seeds": list(range(30)),
        "proportions": np.linspace(0.05, .95, 19),
    }


SEEDS = get_config()["seeds"]


def average_summary(seed_summaries):
    if not seed_summaries:
        return {}

    numeric_keys = [
        "total_societal_value",
        "total_rewards_distributed",
        "successful_projects_count",
        "total_terminations",
        "avg_agent_age",
        "success_rate",
    ]
    averaged = {
        "seed_count": len(seed_summaries),
    }

    for key in numeric_keys:
        values = [float(summary[key]) for summary in seed_summaries if key in summary]
        averaged[f"mean_{key}"] = float(np.mean(values)) if values else 0.0
        averaged[f"std_{key}"] = float(np.std(values)) if len(values) > 1 else 0.0

    policy_keys = sorted({
        policy
        for summary in seed_summaries
        for policy in summary.get("final_populations", {})
    })
    averaged_populations = {}
    averaged_populations_std = {}
    for policy in policy_keys:
        values = [
            float(summary.get("final_populations", {}).get(policy, 0))
            for summary in seed_summaries
        ]
        averaged_populations[policy] = float(np.mean(values)) if values else 0.0
        averaged_populations_std[policy] = float(np.std(values)) if len(values) > 1 else 0.0
    averaged["final_populations_mean"] = averaged_populations
    averaged["final_populations_std"] = averaged_populations_std

    return averaged


def run_experiment_step(args):
    reward_type, proportion, params_dict, seed = args
    adverse_prop = float(proportion)
    remaining = 1.0 - adverse_prop
    other_prop = remaining / 3.0

    policy_dist = {
        "adverse": adverse_prop,
        "careerist": other_prop,
        "orthodox_scientist": other_prop,
        "mass_producer": other_prop,
    }

    print(f"Starting simulation for adverse proportion: {adverse_prop:.2f} (seed {seed})")
    DISTRIBUTION_MODE = "multiply"
    cp = params_dict
    output_prefix = f"tipping_point_adverse_{adverse_prop:.2f}_{reward_type}_seed{seed}"

    result = run_simulation_with_policies(
        n_agents=1_000,
        start_agents=200,
        max_steps=600,
        n_groups=20,
        max_peer_group_size=50,
        policy_distribution=policy_dist,
        output_file_prefix=output_prefix,
        group_policy_homogenous=True,
        reward_type=reward_type,
        distribution_mode=DISTRIBUTION_MODE,
        seed=seed,
        max_rewardless_steps=cp["max_rewardless_steps"],
        acceptance_threshold=cp["acceptance_threshold"],
        novelty_threshold=cp["orthodox_novelty_threshold"],
        prestige_threshold=cp["careerist_prestige_threshold"],
        effort_threshold=cp["mass_producer_effort_threshold"],
        coordination_factor=cp["coordination_factor"],
        continuation_probability=cp["continuation_probability"],
        ratio_group_expansion_depends_on_success=cp.get("ratio_group_expansion_depends_on_success", 0.5),
        verbose=False,
    )

    stats = result["final_stats"]
    summary = {
        "seed": seed,
        "adverse_proportion": adverse_prop,
        "total_societal_value": stats.get("total_societal_value", 0),
        "total_rewards_distributed": stats.get("total_rewards_distributed", 0),
        "successful_projects_count": stats.get("successful_projects", 0),
        "total_terminations": stats.get("total_terminations", 0),
        "avg_agent_age": stats.get("observation_aggregates", {}).get("avg_age", 0),
        "final_populations": result["policy_populations"],
        "success_rate": stats.get("success_rate", 0),
    }

    print(f"Finished simulation for adverse proportion: {adverse_prop:.2f} (seed {seed})")
    return summary


def main():
    config = get_config()
    seeds = config["seeds"]
    proportions = config["proportions"]
    print(proportions)

    # Use "all" calibrated params as specified in run_policy_simulation.py
    REWARD_TYPE = "all"
    # try different reward types and make all plots
    cp = {k: v for k, v in CALIBRATED_PARAMS[REWARD_TYPE]}
    cp["ratio_group_expansion_depends_on_success"] = 0.5
    for reward_type in ["reputation", "raw_pubcount", "h_index"]:
        tasks = [(reward_type, p, cp, seed) for p in proportions for seed in seeds]
        results = []

        print(
            f"Starting tipping point experiment with {len(tasks)} runs across seeds {min(seeds)}-{max(seeds)}..."
        )
        with ProcessPoolExecutor(max_workers=min(41, len(tasks))) as executor:
            results = list(executor.map(run_experiment_step, tasks))

        grouped_results = {}
        for summary in results:
            grouped_results.setdefault(summary["adverse_proportion"], []).append(summary)

        averaged_results = []
        for proportion in proportions:
            seed_summaries = grouped_results.get(float(proportion), [])
            if not seed_summaries:
                continue
            averaged_summary = average_summary(seed_summaries)
            averaged_summary["adverse_proportion"] = float(proportion)
            averaged_results.append(averaged_summary)

        averaged_results.sort(key=lambda x: x["adverse_proportion"])

        output_file = f"results/tipping_point_experiment_{reward_type}_results.json"
        with open(output_file, "w") as f:
            json.dump(averaged_results, f, indent=2)

        print(f"Experiment completed. Average results across seeds 0-9 saved to {output_file}")
        for entry in averaged_results:
            print(
                f"adverse_proportion={entry['adverse_proportion']:.2f} | "
                f"avg_total_societal_value={entry['mean_total_societal_value']:.3f} | "
                f"avg_success_rate={entry['mean_success_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
    # cp = {k: v for k, v in CALIBRATED_PARAMS["all"]}
    # s = run_experiment_step(("all", 0.5 , cp, 0))
    # print(s)