import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from run_policy_simulation import run_simulation_with_policies, CALIBRATED_PARAMS

def run_experiment_step(args):
    proportion, params_dict = args
    adverse_prop = float(proportion)
    remaining = 1.0 - adverse_prop
    other_prop = remaining / 3.0
    
    policy_dist = {
        "adverse": adverse_prop,
        "careerist": other_prop,
        "orthodox_scientist": other_prop,
        "mass_producer": other_prop
    }
    
    print(f"Starting simulation for adverse proportion: {adverse_prop:.2f}")
    
    # Run the simulation using calibrated parameters for "all" (reputation+hindex+pubcount combined if available, 
    # but the user was using REWARD_TYPE="all" with DISTRIBUTION_MODE="multiply")
    # Actually we should use the REWARD_TYPE="all" and DISTRIBUTION_MODE="multiply" as they set in their script
    REWARD_TYPE = "all"
    DISTRIBUTION_MODE = "multiply"
    
    # Fallback to defaults if "all" isn't calibrated. In their code, "all" was in CALIBRATED_PARAMS.
    cp = params_dict
    
    output_prefix = f"tipping_point_adverse_{adverse_prop:.2f}_{REWARD_TYPE}"
    
    result = run_simulation_with_policies(
        n_agents=3000,
        start_agents=200,
        max_steps=600,
        n_groups=20,
        max_peer_group_size=150,
        policy_distribution=policy_dist,
        output_file_prefix=output_prefix,
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
        verbose=False
    )
    
    stats = result["final_stats"]
    
    summary = {
        "adverse_proportion": adverse_prop,
        "total_societal_value": stats.get("total_societal_value", 0),
        "total_rewards_distributed": stats.get("total_rewards_distributed", 0),
        "successful_projects_count": stats.get("successful_projects", 0),
        "total_terminations": stats.get("total_terminations", 0),
        "avg_agent_age": stats.get("observation_aggregates", {}).get("avg_age", 0),
        "final_populations": result["policy_populations"],
        "success_rate": stats.get("success_rate", 0)
    }
    
    print(f"Finished simulation for adverse proportion: {adverse_prop:.2f}")
    return summary

def main():
    proportions = np.linspace(0.0, 1.0, 21) # 0.0, 0.05, 0.10, ..., 1.0
    
    # Use "all" calibrated params as specified in run_policy_simulation.py
    REWARD_TYPE = "reputation"
    # try different reward types and make all plots
    cp = {k: v for k, v in CALIBRATED_PARAMS[REWARD_TYPE]}
        
    tasks = [(p, cp) for p in proportions]
    
    results = []
    # Run in parallel
    print(f"Starting tipping point experiment with {len(tasks)} runs...")
    with ProcessPoolExecutor(max_workers=21) as executor:
        results = list(executor.map(run_experiment_step, tasks))
        
    # Sort results by proportion just in case
    results.sort(key=lambda x: x["adverse_proportion"])
    
    # Save to JSON
    output_file = f"log/tipping_point_experiment_{REWARD_TYPE}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiment completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
