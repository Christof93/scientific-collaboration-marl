import json
import os
import numpy as np
import pickle
from typing import Dict, List, Any

# Mock class to reconstruct peer groups exactly like the environment
class PeerGroupReconstructor:
    def __init__(self, n_agents=2000, n_groups=20, seed=42):
        self.n_agents = n_agents
        self.n_groups = n_groups
        np.random.seed(seed)
        
        # 1. Basic groups
        self.peer_groups = [[] for _ in range(self.n_groups)]
        for i in range(self.n_agents):
            self.peer_groups[i % self.n_groups].append(i)
            
        # 2. Connections (deterministic if seed is 42)
        # We need active_agents for _connect_peer_groups, but wait...
        # In the real env, _connect_peer_groups is called in __init__.
        # Let's check when it's called.
        
def get_peer_groups(n_agents=2000, n_groups=20, seed=42):
    # Replicating PeerGroupEnvironment._init_peer_groups
    groups = [[] for _ in range(n_groups)]
    for i in range(n_agents):
        groups[i % n_groups].append(i)
    
    # Replicating _connect_peer_groups
    # Note: _connect_peer_groups uses np.random.permutation and np.random.choice
    # BUT it also checks self.active_agents!
    # If active_agents changes over time, then _connect_peer_groups (if called repeatedly)
    # would be dynamic. But it's usually called once.
    # Wait, let's check PeerGroupEnvironment.reset().
    return groups

def replay(prefix, steps_to_show=10):
    summary_path = f"{prefix}_summary.json"
    obs_path = f"log/{prefix}_observations.jsonl"
    act_path = f"log/{prefix}_actions.jsonl"

    if not os.path.exists(summary_path):
        # Fallback to current dir or check log/
        if os.path.exists(f"log/{summary_path}"):
            summary_path = f"log/{summary_path}"
        else:
            print(f"Could not find {summary_path}")
            return

    with open(summary_path) as f:
        summary = json.load(f)
    
    agent_policies = summary.get("agent_policies", [])
    n_agents = len(agent_policies)
    n_groups = 20 # Default from calibrate.py
    
    # Reconstruct groups
    # Since _connect_peer_groups is stochastic and uses active_agents,
    # and we don't have the exact state, we'll assume the primary groups for now.
    # Most agents stay in their primary group.
    groups = [[] for _ in range(n_groups)]
    for i in range(n_agents):
        groups[i % n_groups].append(i)

    # Find top agents by final reputation
    print("Finding top agents...")
    final_reputations = {}
    last_line = None
    try:
        with open(obs_path) as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line:
            obs_row = json.loads(last_line)
            for agent_id, data in obs_row.items():
                if data:
                    final_reputations[agent_id] = data["observation"]["accumulated_rewards"][0]
    except Exception as e:
        print(f"Error reading reputations: {e}")
        return

    archetypes = ["careerist", "orthodox_scientist", "mass_producer"]
    top_agents = {}
    for arch in archetypes:
        best_id = None
        max_rep = -1.0
        for i, policy in enumerate(agent_policies):
            if policy == arch:
                aid = f"agent_{i}"
                rep = final_reputations.get(aid, 0.0)
                if rep > max_rep:
                    max_rep = rep
                    best_id = aid
        top_agents[arch] = best_id

    print(f"Top agents identified:")
    for arch, aid in top_agents.items():
        print(f"  {arch}: {aid} (Final Reputation: {final_reputations.get(aid, 0):.2f})")

    # Colors
    C_END = "\033[0m"
    C_BOLD = "\033[1m"
    C_CYAN = "\033[36m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_MAGENTA = "\033[35m"

    arch_colors = {
        "careerist": C_MAGENTA,
        "orthodox_scientist": C_CYAN,
        "mass_producer": C_YELLOW
    }

    # Replay
    print(f"\nStarting replay (first {steps_to_show} steps)...")
    try:
        with open(obs_path) as f_obs, open(act_path) as f_act:
            step = 0
            for line_obs, line_act in zip(f_obs, f_act):
                if step >= steps_to_show:
                    break
                    
                obs_row = json.loads(line_obs)
                act_row = json.loads(line_act)
                
                # Check if at least one top agent is active in this step
                active_step_agents = [aid for aid in top_agents.values() if aid in obs_row and obs_row[aid] is not None]
                
                if not active_step_agents:
                    step += 1
                    continue

                print(f"\n{C_BOLD}{'='*100}{C_END}")
                print(f"{C_BOLD}STEP {step}{C_END}")
                print(f"{C_BOLD}{'='*100}{C_END}")

                for arch, aid in top_agents.items():
                    if aid not in active_step_agents:
                        continue
                    
                    inner_obs = obs_row[aid]["observation"]
                    act = act_row[aid]
                    
                    rep = inner_obs["accumulated_rewards"][0]
                    centroid = inner_obs["self_centroid"][0]
                    running = inner_obs.get("running_projects", {})
                    opps = inner_obs.get("project_opportunities", {})
                    
                    color = arch_colors.get(arch, "")
                    print(f"\n{C_BOLD}Actor: {color}{aid}{C_END} ({color}{arch}{C_END})")
                    print(f"  {C_BOLD}Reputation:{C_END} {rep:6.2f} | {C_BOLD}Running:{C_END} {len(running)} | {C_BOLD}Opps:{C_END} {len(opps)} | {C_BOLD}Topic:{C_END} ({centroid[0]:.2f}, {centroid[1]:.2f})")
                    
                    # Action formatting
                    act_str = f"Effort: {act.get('put_effort')}"
                    if act.get('choose_project'):
                        act_str += f" | {C_GREEN}START PROJECT{C_END}"
                    
                    collabs = act.get('collaborate_with', [])
                    active_collabs = sum(collabs)
                    if active_collabs > 0:
                        act_str += f" | Collaborate with {active_collabs} peers"
                    
                    print(f"  {C_BOLD}Action:{C_END} {act_str}")
                    
                    # Peers
                    peer_active_mask = inner_obs["peer_group"]
                    peer_reps = inner_obs["peer_reputation"]
                    peer_cents = inner_obs["peer_centroids"]
                    
                    agent_idx = int(aid.split("_")[1])
                    peer_ids = groups[agent_idx % n_groups]
                    
                    print(f"  {C_BOLD}Peers (Active Top 5):{C_END}")
                    active_peers = []
                    for i in range(min(len(peer_active_mask), len(peer_ids))):
                        if peer_active_mask[i]:
                            p_idx = peer_ids[i]
                            p_arch = agent_policies[p_idx]
                            p_rep = peer_reps[i]
                            p_cent = peer_cents[i]
                            active_peers.append((p_idx, p_arch, p_rep, p_cent))
                    
                    # Sort active peers by reputation
                    active_peers.sort(key=lambda x: x[2], reverse=True)
                    for p_idx, p_arch, p_rep, p_cent in active_peers[:5]:
                        p_color = arch_colors.get(p_arch, "")
                        print(f"    - agent_{p_idx:4} ({p_color}{p_arch:18}{C_END}) | Rep: {p_rep:6.2f} | Topic: ({p_cent[0]:5.2f}, {p_cent[1]:5.2f})")
                
                step += 1
    except Exception as e:
        print(f"Error during replay: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="balanced_multiply_seed42")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    replay(args.prefix, args.steps)
