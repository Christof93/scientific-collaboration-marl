import numpy as np
from typing import Dict, List, Any, Optional

class RewardManager:
    """
    Manages reward distribution logic for the PeerGroupEnvironment.
    Modularizes different reward mechanisms.
    """
    def __init__(self, env):
        self.env = env
        self.reward_function_name = env.reward_function_name
        self.prev_h_indexes = np.zeros(env.n_agents, dtype=np.int16)

    def reset(self):
        """Called during environment reset."""
        self.prev_h_indexes = self.env.agent_h_indexes.copy()

    def distribute_project_reward(self, project: Any, reward: float):
        """
        Distributes rewards when a project is completed (accepted or failed).
        
        Args:
            project: The project object that was completed.
            reward: The calculated reward based on project quality and threshold.
        """
        if self.reward_function_name == "multiply":
            self._distribute_multiply(project, reward)
        elif self.reward_function_name == "evenly":
            self._distribute_evenly(project, reward)
        elif self.reward_function_name == "by_effort":
            self._distribute_by_effort(project, reward)
        elif self.reward_function_name == "publications":
            self._distribute_publications(project, reward)
        elif self.reward_function_name == "h_index_diff":
            # Immediate project reward is handled via h-index delta in step_end
            # But we still need to cleanup the project state
            self._cleanup_project(project, reward)
        else:
            # Fallback/Default (multiply)
            self._distribute_multiply(project, reward)

    def apply_step_rewards(self):
        """
        Applied at the end of every step to handle rewards based on step-over-step deltas.
        Currently handles 'h_index_diff'.
        """
        if self.reward_function_name == "h_index_diff":
            # Reward is delta in h-index for ACTIVE agents
            current_h = self.env.agent_h_indexes
            delta = current_h - self.prev_h_indexes
            
            for i in range(self.env.n_agents):
                # Only reward active agents as per user requirement
                if self.env.active_agents[i] == 1 and delta[i] > 0:
                    reward_val = float(delta[i])
                    self.env.agent_rewards[i, self.env.timestep] += reward_val
                    self.env.rewards[f"agent_{i}"] += reward_val
            
            # Update previous state for next step
            self.prev_h_indexes = current_h.copy()

    def _cleanup_project(self, p, reward):
        """Standard state update for finished projects without immediate reward distribution."""
        for idx in p.contributors:
            self.env.remove_active_project(idx, p.project_id)
            if reward > 0:
                self.env.agent_successful_projects[idx].append(p.project_id)
            self.env.agent_completed_projects[idx] += 1

    def _distribute_multiply(self, p, reward):
        """Each contributor gets the full project reward."""
        for idx in p.contributors:
            self.env.remove_active_project(idx, p.project_id)
            if reward > 0:
                self.env.agent_successful_projects[idx].append(p.project_id)
                self.env.agent_rewards[idx, self.env.timestep] += reward
                self.env.rewards[f"agent_{idx}"] += float(reward)
            self.env.agent_completed_projects[idx] += 1

    def _distribute_evenly(self, p, reward):
        """Project reward is split evenly among contributors."""
        n = len(p.contributors)
        share = reward / n if n > 0 else 0
        for idx in p.contributors:
            self.env.remove_active_project(idx, p.project_id)
            if reward > 0:
                self.env.agent_successful_projects[idx].append(p.project_id)
                self.env.agent_rewards[idx, self.env.timestep] += share
                self.env.rewards[f"agent_{idx}"] += float(share)
            self.env.agent_completed_projects[idx] += 1

    def _distribute_by_effort(self, p, reward):
        """Project reward is split proportionally to relative effort."""
        efforts = [self.env.agent_project_effort[c][p.project_id] for c in p.contributors]
        max_effort = max(efforts) if efforts else 0
        for idx in p.contributors:
            effort = self.env.agent_project_effort[idx][p.project_id]
            rel_effort = (effort / max_effort if max_effort > 0 else 1 / len(p.contributors))
            self.env.remove_active_project(idx, p.project_id)
            if reward > 0:
                self.env.agent_successful_projects[idx].append(p.project_id)
                self.env.agent_rewards[idx, self.env.timestep] += reward * rel_effort
                self.env.rewards[f"agent_{idx}"] += float(reward * rel_effort)
            self.env.agent_completed_projects[idx] += 1

    def _distribute_publications(self, p, reward):
        """Purely based on number of publications. Each contributor gets 1 reward if accepted."""
        for idx in p.contributors:
            self.env.remove_active_project(idx, p.project_id)
            if reward > 0:
                pub_reward = 1.0
                self.env.agent_successful_projects[idx].append(p.project_id)
                self.env.agent_rewards[idx, self.env.timestep] += pub_reward
                self.env.rewards[f"agent_{idx}"] += pub_reward
            self.env.agent_completed_projects[idx] += 1
