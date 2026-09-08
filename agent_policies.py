"""
Agent Policy Functions for Peer Group Environment

This module contains three different agent policy functions that can be used
instead of random sampling in the peer group environment simulation.

1. careerist: Chooses projects with high potential reward
2. orthodox_scientist: Chooses projects with good fit
3. mass_producer: Chooses projects with low effort and short completion time
"""
from collections import Counter, defaultdict
from itertools import zip_longest
from typing import Any, Dict, List

import numpy as np
from env.area import Area


def _mask_allowed(mask_arr: Any, idx: int) -> bool:
    if mask_arr is None:
        return True
    try:
        return mask_arr[idx] > 0
    except Exception:
        return False


def _emergency_continue_any(running_projects: Dict[str, Any]) -> bool:
    """
    Determine if there is too much workload to begin new projects.
    Uses realistic peer-fit-based effort estimates, accounts for attention limits, 
    and robustly handles dead/contributor-less teams.
    """
    for proj in running_projects.values():
        required = proj.get("required_effort")[0]
        current = proj.get("current_effort")[0]
        
        # If project is already fully completed, it doesn't block us
        if current >= required:
            continue
            
        contributors = sum(proj.get("contributors", []))
        
        # If there are no active contributors but the project still needs effort,
        # it is mathematically impossible to complete. Immediately block new projects.
        if contributors < 1:
            return True
            
        time_left = proj.get("time_left")[0]
        
        # Instead of assuming 1.0 effort per step, we sum the actual peer_fit
        # of the active contributors (non-contributors have fit set to 0.0).
        max_effort_per_step = sum(proj.get("peer_fit", []))
        
        # Assuming agents split their attention among their active projects.
        own_active_count = max(len(running_projects), 1)
        estimated_attention_factor = 1.0 / own_active_count
        
        expected_future_effort = time_left * max_effort_per_step * estimated_attention_factor
        
        # If the expected future effort is strictly smaller than remaining required effort,
        # the project is on track to fail -> return True to focus on current work.
        if expected_future_effort < (required - current):
            return True
            
    return False


def _should_block_new_project(running_projects: Dict[str, Any]) -> bool:
    """
    Determine if we should block choosing a new project due to existing workload.
    """
    if _emergency_continue_any(running_projects):
        return True
        
    active_projects = []
    for proj in running_projects.values():
        time_left = proj.get("time_left")[0]
        if time_left <= 0:
            continue
        active_projects.append(proj)
        
    if not active_projects:
        return False
        
    total_ratio = 0.0
    for proj in active_projects:
        required = proj.get("required_effort")[0]
        current = proj.get("current_effort")[0]
        time_left = proj.get("time_left")[0]
        
        contributors = sum(proj.get("contributors", []))
        n_contributors = max(contributors, 1)
        time_left = max(time_left, 1)
        
        ratio = ((required - current) / n_contributors) / time_left
        total_ratio += ratio
        
    mean_ratio = total_ratio / len(active_projects)
    return mean_ratio >= 1.0


def _select_effort_closest_deadline_under_required(
    running_projects: Dict[str, Any], put_effort_mask: np.ndarray
) -> int:
    """
    choose the running project with smallest time_left that still needs effort
    """
    candidates: List[tuple] = []
    for slot_idx, proj in enumerate(running_projects.values()):
        time_left = proj["time_left"][0]
        if time_left <= 0:
            continue
        if proj["current_effort"][0] < proj["required_effort"][0] and _mask_allowed(
            put_effort_mask, slot_idx + 1
        ):
            candidates.append((slot_idx + 1, time_left))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    return 0


def _select_effort_best_fit_or_threshold(
    running_projects: Dict[str, Any],
    put_effort_mask: np.ndarray,
    threshold_ratio: float = 0.9,
) -> int:
    # If any project is above threshold, immediately work on it; else choose by best peer_fit
    candidates: List[tuple] = []
    for slot_idx, proj in enumerate(running_projects.values()):
        required = proj["required_effort"][0]
        threshold = required * threshold_ratio
        if (
            proj["current_effort"][0] >= threshold
            and proj["current_effort"][0] < required
            and _mask_allowed(put_effort_mask, slot_idx + 1)
        ):
            return slot_idx + 1
        if _mask_allowed(put_effort_mask, slot_idx + 1):
            fit = float(np.sum(proj["peer_fit"])) if len(proj["peer_fit"]) > 0 else 0.0
            candidates.append((slot_idx + 1, fit))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return 0


def maximally_collaborative_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    **kwargs,
) -> Dict[str, Any]:
    collaborate_mask = action_mask.get("collaborate_with")
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )
    return {
        "choose_project": 1,
        "collaborate_with": collaborate_mask,
        "put_effort": put_effort,
    }


def careerist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    prestige_threshold: float = 0.6,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    current = observation.get("running_projects", {})
    if _should_block_new_project(current):
        chosen_project = 0
    else:
        choose_project_mask = action_mask.get("choose_project")
        # print(project_opportunities)
        if len(project_opportunities) > 0:
            opp = project_opportunities[0]
            meets = (
                opp is not None
                and float(opp.get("prestige")[0]) >= prestige_threshold
                and _mask_allowed(choose_project_mask, 1)
            )
            chosen_project = 1 if meets else 0
        else:
            chosen_project = 0

    # Collaboration: active peers with above-average reputation
    peer_reputation = np.array(observation.get("peer_reputation", []), dtype=np.float32)
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")
    own_reputation = observation.get("accumulated_rewards")[0]
    avg_rep = (
        float(peer_reputation[peer_group_active == 1].mean())
        if peer_group_active.sum() > 0
        else 0.0
    )
    # collaborate only with higher than average reputation
    higher_reputation = peer_reputation >= own_reputation
    if higher_reputation.sum() >= 1:
        desired = ((peer_reputation >= avg_rep).astype(np.int8)) * (
            peer_group_active > 0
        ).astype(np.int8)
    # if highest reputation collaborate with all
    else:
        desired = (peer_group_active > 0).astype(np.int8)
    collaborate_with = (desired > 0) & (collaborate_mask > 0)
    collaborate_with = collaborate_with.astype(np.int8)

    # Effort: project closest to deadline still under required_effort
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def orthodox_scientist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    novelty_threshold: float = 0.4,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    current = observation.get("running_projects", {})
    if _should_block_new_project(current):
        chosen_project = 0
    else:
        choose_project_mask = action_mask.get("choose_project")
        if len(project_opportunities) > 0:
            opp = project_opportunities[0]
            meets = (
                opp is not None
                and float(opp.get("novelty")[0]) <= novelty_threshold
                and _mask_allowed(choose_project_mask, 1)
            )
            chosen_project = 1 if meets else 0
        else:
            chosen_project = 0

    # Collaboration: only collaborate with agents who are close in topic centroid
    peer_group_active = np.array(observation.get("peer_group"), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")

    # Get agent's own topic centroid
    own_centroid = np.array(observation.get("self_centroid")[0])
    peer_centroids = np.array(observation.get("peer_centroids"))

    # Calculate distances to all peers
    if len(peer_centroids) > 0 and len(own_centroid) > 0:
        distances = Area.distance(own_centroid, peer_centroids)
        # Collaborate only with peers closer than average distance
        avg_distance = np.mean(distances)
        close_peers = (distances < avg_distance) & (peer_group_active > 0)
        collaborate_with = (close_peers & (collaborate_mask > 0)).astype(np.int8)
    else:
        # Fallback: collaborate with all active peers if centroids not available
        collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
            np.int8
        )

    # Effort: best fitting active project or above 90% threshold
    running_projects = observation.get("running_projects", {})
    put_effort_mask = action_mask.get("put_effort")
    put_effort = _select_effort_best_fit_or_threshold(
        running_projects, put_effort_mask, threshold_ratio=0.9
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def mass_producer_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    effort_threshold: int = 22,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    current = observation.get("running_projects", {})
    choose_project_mask = action_mask.get("choose_project")
    if _should_block_new_project(current):
        chosen_project = 0
    elif len(project_opportunities) > 0:
        opp = project_opportunities[0]
        if opp is not None and opp.get("required_effort")[0] <= effort_threshold and _mask_allowed(choose_project_mask, 1):
            chosen_project = 1
        else:
            chosen_project = 0
        # if opp is not None and _mask_allowed(choose_project_mask, 1):
        #     effort = float(opp.get("required_effort")[0])
        #     time_w = float(opp.get("time_window")[0])
        #     prestige = float(opp.get("prestige")[0])
        #     eff = prestige / (effort * time_w) if effort > 0 and time_w > 0 else 0.0
        #     chosen_project = 1 if eff > 0 else 0
        # else:
        #     chosen_project = 0
    else:
        chosen_project = 0

    # Collaborate with all active peers within mask
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")
    collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
        np.int8
    )
    # if sum(collaborate_with) == 0:
    #     breakpoint()

    # Effort: project closest to deadline under required_effort
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def random_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    **kwargs,
) -> Dict[str, Any]:
    """Takes random actions allowed by the given mask."""
    # Choose a random project from allowed
    choose_mask = action_mask.get("choose_project")
    allowed_projects = np.where(choose_mask > 0)[0]
    chosen_project = (
        np.random.choice(allowed_projects) if len(allowed_projects) > 0 else 0
    )

    # Randomly collaborate with allowed peers
    collaborate_mask = action_mask.get("collaborate_with")
    collaborate_with = np.zeros_like(collaborate_mask)
    allowed_peers = np.where(collaborate_mask > 0)[0]
    if len(allowed_peers) > 0:
        collaborate_with[allowed_peers] = np.random.randint(
            0, 2, size=len(allowed_peers)
        )

    # Choose a random project to put effort into
    put_effort_mask = action_mask.get("put_effort")
    allowed_effort = np.where(put_effort_mask > 0)[0]
    put_effort = np.random.choice(allowed_effort) if len(allowed_effort) > 0 else 0

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def adverse_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    prestige_threshold: float = 0.4,
    **kwargs,
) -> Dict[str, Any]:
    """Adverse tactic: never spending effort, always collaborating, taking only low prestige projects."""
    # Taking only low prestige projects
    project_opportunities = list(observation.get("project_opportunities").values())
    choose_project_mask = action_mask.get("choose_project")
    chosen_project = 0

    # Check all opportunities
    for i, opp in enumerate(project_opportunities):
        if opp is not None and _mask_allowed(choose_project_mask, i + 1):
            if float(opp.get("prestige")[0]) < prestige_threshold:
                chosen_project = i + 1
                break

    # Always collaborating
    collaborate_mask = action_mask.get("collaborate_with")
    collaborate_with = (collaborate_mask > 0).astype(np.int8)

    # Never spending effort
    put_effort = 0

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def do_nothing_policy(_: Dict[str, Any], action_mask: Dict[str, np.ndarray]):
    return {
        "choose_project": 0,
        "collaborate_with": np.zeros_like(action_mask["collaborate_with"]),
        "put_effort": 0,
    }


def interleave(lists):
    return [elem for group in zip_longest(*lists) for elem in group if elem is not None]


def get_policy_function(policy_name: str):
    policies = {
        "careerist": careerist_policy,
        "orthodox_scientist": orthodox_scientist_policy,
        "mass_producer": mass_producer_policy,
        "maximally_collaborative": maximally_collaborative_policy,
        "random": random_policy,
        "adverse": adverse_policy,
    }
    if policy_name not in policies:
        raise ValueError(
            f"Unknown policy: {policy_name}. Available policies: {list(policies.keys())}"
        )
    return policies[policy_name]


def create_mixed_policy_population(
    n_agents: int, n_groups: int, policy_distribution: Dict[str, float] = None, seed=None
) -> List[str]:
    if seed is not None:
        np.random.seed(seed)
    if policy_distribution is None:
        policy_distribution = {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        }
    group_size = n_agents // n_groups
    total_proportion = sum(policy_distribution.values())
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")
    agent_policies = []
    for policy_name, proportion in policy_distribution.items():
        n_policy_agents = int(n_agents * proportion)
        agent_policies.extend([policy_name] * n_policy_agents)
    while len(agent_policies) < n_agents:
        agent_policies.append(policy_name)
    np.random.shuffle(agent_policies)

    agent_to_group = np.full(n_agents, -1, dtype=int)

    for agent_id in range(n_agents):
        agent_to_group[agent_id] = agent_id % n_groups
    
    return agent_to_group.tolist(), agent_policies


# def create_per_group_policy_population(
#     n_agents: int, policy_distribution: Dict[str, float] = None
# ) -> List[str]:
#     if policy_distribution is None:
#         policy_distribution = {
#             "careerist": 1 / 3,
#             "orthodox_scientist": 1 / 3,
#             "mass_producer": 1 / 3,
#         }
#     total_proportion = sum(policy_distribution.values())
#     if abs(total_proportion - 1.0) > 1e-6:
#         raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")
#     policy_groups = []
#     for policy_name, proportion in policy_distribution.items():
#         if proportion > 0:
#             n_policy_agents = int(n_agents * proportion)
#             policy_groups.append([policy_name] * n_policy_agents)
#     while sum([len(group) for group in policy_groups]) < n_agents:
#         policy_groups[-1].append(list(policy_distribution.keys())[0])
#     return interleave(policy_groups)

def create_per_group_policy_population(
        n_agents: int,
        n_groups: int,
        policy_distribution: dict[str, float],
        seed: int | None = None,
    ) -> tuple[list[int], list[str]]:
        """
        Assign each agent to a peer group such that:
        - every peer group has the same size
        - every peer group is homogeneous with respect to policy
        - the number of groups per policy follows policy_distribution
        - agents are randomly distributed across groups

        Returns:
            agent_to_group:
                List where agent_to_group[agent_id] is the group ID.

            group_policies:
                List where group_policies[group_id] is the policy of that group.
        """

        if n_agents % n_groups != 0:
            raise ValueError(
                f"n_agents ({n_agents}) must be divisible by "
                f"n_groups ({n_groups}) for equal-sized groups."
            )

        if abs(sum(policy_distribution.values()) - 1.0) > 1e-6:
            raise ValueError("Policy distribution must sum to 1.0")

        rng = np.random.default_rng(seed)

        max_group_size = n_agents // n_groups

        policies = list(policy_distribution.keys())
        proportions = np.array(
            [policy_distribution[p] for p in policies],
            dtype=float,
        )

        # agents per group
        agent_policies = []
        for policy_name, proportion in policy_distribution.items():
            n_policy_agents = int(n_agents * proportion)
            agent_policies.extend([policy_name] * n_policy_agents)
        while len(agent_policies) < n_agents:
            agent_policies.append(policy_name)

        true_agents_per_policy = dict(zip(*np.unique(agent_policies, return_counts=True)))
        group_counts = np.ones(len(policies), dtype=int)

        # Remaining groups to distribute
        remaining_groups = n_groups - len(policies)
        if remaining_groups > 0:
            additional_exact = proportions * remaining_groups

            additional_counts = np.floor(additional_exact).astype(int)

            group_counts += additional_counts

            remaining = remaining_groups - additional_counts.sum()

            fractional = additional_exact - additional_counts
            order = np.argsort(-fractional)

            for i in order[:remaining]:
                group_counts[i] += 1
                
        assert group_counts.sum() == n_groups
        agent_to_group = np.full(n_agents, -1, dtype=int)

        currently_assigned = 0
        current_group = 0
        for policy_i, policy in enumerate(policies):
            n_policy_agents = true_agents_per_policy[policy]
            n_policy_groups = group_counts[policy_i]
            current_group_size = n_policy_agents // n_policy_groups
            
            assigned = 0
            for g in range(n_policy_groups):
                start = currently_assigned + current_group_size * g
                end = start + current_group_size
                agent_to_group[start:end] = current_group
                current_group += 1
                assigned += current_group_size
            remaining = n_policy_agents - assigned
            agent_to_group[end:end+remaining] = current_group - 1
            currently_assigned += assigned + remaining
        assert np.all(agent_to_group >= 0)
        assert np.all(agent_to_group < n_groups)
        return agent_to_group.tolist(), agent_policies

def validate_proportions(agent_to_group, agent_policies):
    count = defaultdict(int)
    ptype = defaultdict(set)
    for i, policy in enumerate(agent_policies):
        count[agent_to_group[i]] += 1
        ptype[agent_to_group[i]] |= {policy}
    for g, c in count.items():
        print(g, c, ptype[g])
    assert sum(count.values()) == 3000
    print()
    assert len(ptype) == 20
    assert max([len(v) for v in ptype.values()]) == 1

    print()

if __name__ == "__main__":
    # Keep minimal manual check without noisy prints
    agent_to_group, agent_policies = create_per_group_policy_population(3000, n_groups=20, policy_distribution = {
            "careerist": 0.25,
            "orthodox_scientist": 0.75/2,
            "mass_producer": 0.75/2,
        }
    )
    validate_proportions(agent_to_group, agent_policies)

    agent_to_group, agent_policies = create_mixed_policy_population(3000, n_groups=20, policy_distribution = {
                "careerist": 0.25,
                "orthodox_scientist": 0.75/2,
                "mass_producer": 0.75/2,
            }
        )
    count = defaultdict(int)
    ptype = defaultdict(set)
    for i, policy in enumerate(agent_policies):
        count[agent_to_group[i]] += 1
        ptype[agent_to_group[i]] |= {policy}
        
    for g, c in count.items():
        print(g, c, ptype[g])
    print(Counter(agent_policies).most_common())
    