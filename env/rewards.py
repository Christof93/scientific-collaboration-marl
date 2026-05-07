import numpy as np
from typing import Dict, List, Any, Optional

class RewardManager:
    """
    Manages reward distribution logic for the PeerGroupEnvironment.
    Decouples Reward Source (reputation, raw_pubcount, h_index) 
    from Distribution Mode (multiply, evenly, by_effort).
    """
    def __init__(self, env):
        self.env = env
        self.reward_type = env.reward_type
        self.distribution_mode = env.distribution_mode
        self.prev_h_indexes = np.zeros(env.n_agents, dtype=np.int16)

    def reset(self):
        """Called during environment reset."""
        self.prev_h_indexes = np.zeros(self.env.n_agents, dtype=np.int16)

    def distribute_project_reward(self, project: Any, quality_reward: float):
        """
        Calculates the base reward based on reward_type and distributes it 
        according to distribution_mode.
        
        Args:
            project: The project object that was completed.
            quality_reward: The calculated reward based on project quality and threshold.
        """
        # 1. Determine Success
        is_successful = quality_reward > 0
        if is_successful:
            for idx in project.contributors:
                if project.project_id not in self.env.agent_successful_projects[idx]:
                    self.env.agent_successful_projects[idx].append(project.project_id)

        # 2. Determine Base Reward
        base_reward = 0.0
        if self.reward_type == "reputation":
            base_reward = quality_reward
        elif self.reward_type == "raw_pubcount":
            base_reward = 1.0 if is_successful else 0.0
        elif self.reward_type == "h_index":
            # h_index rewards are handled in apply_step_rewards
            base_reward = 0.0

        # 3. Distribute Reward (if any) and Cleanup
        if base_reward > 0:
            self._apply_distribution(project, base_reward)
        else:
            self._cleanup_project(project)

    def apply_step_rewards(self):
        """
        Applied at the end of every step to handle rewards based on step-over-step deltas.
        Used for 'h_index' reward type.
        """
        if self.reward_type == "h_index":
            current_h = self.env.agent_h_indexes
            delta = current_h - self.prev_h_indexes
            for i in range(self.env.n_agents):
                if delta[i] > 0:
                    reward_val = float(delta[i])
                    self.env.agent_rewards[i, self.env.timestep] = reward_val
                    self.env.rewards[f"agent_{i}"] += reward_val
            
            self.prev_h_indexes = current_h.copy()

    def _cleanup_project(self, p):
        """Standard state update for finished projects (cleanup only)."""
        for idx in p.contributors:
            self.env.remove_active_project(idx, p.project_id)
            # Even if no reward, we track completion
            self.env.agent_completed_projects[idx] += 1

    def _apply_distribution(self, p, total_reward):
        """Shared logic to distribute a total reward pool among project contributors."""
        # Clean up slots first
        self._cleanup_project(p)

        # Apply distribution
        if self.distribution_mode == "multiply":
            for idx in p.contributors:
                self.env.agent_rewards[idx, self.env.timestep] += total_reward
                self.env.rewards[f"agent_{idx}"] += float(total_reward)

        elif self.distribution_mode == "evenly":
            n = len(p.contributors)
            share = total_reward / n if n > 0 else 0
            for idx in p.contributors:
                self.env.agent_rewards[idx, self.env.timestep] += share
                self.env.rewards[f"agent_{idx}"] += float(share)

        elif self.distribution_mode == "by_effort":
            efforts = [self.env.agent_project_effort[c][p.project_id] for c in p.contributors]
            total_effort = sum(efforts) if efforts else 0
            for idx in p.contributors:
                effort = self.env.agent_project_effort[idx][p.project_id]
                rel_effort = (effort / total_effort if total_effort > 0 else 1 / len(p.contributors))
                share = total_reward * rel_effort
                self.env.agent_rewards[idx, self.env.timestep] += share
                self.env.rewards[f"agent_{idx}"] += float(share)
