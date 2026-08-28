# Scientific Knowledge Production Simulation

A multi-agent reinforcement-learning simulation of scientific knowledge production. Agents with different behavioural archetypes (careerist, orthodox scientist, mass producer) form peer groups, collaborate on projects, and accumulate rewards under configurable incentive structures.

## Prerequisites
- Python 3.9+
- Use a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
├── env/                          # Core simulation environment
│   ├── peer_group_environment.py # PettingZoo multi-agent environment
│   ├── project.py                # Project class (effort, quality, evaluation)
│   ├── area.py                   # Topic area & knowledge kene representation
│   ├── rewards.py                # Reward functions (reputation, h-index, pubcount)
│   └── utils.py                  # Shared utilities
├── agent_policies.py             # Agent decision-making policies
├── run_policy_simulation.py      # Main simulation runner
├── calibrate.py                  # Bayesian calibration & Sobol sensitivity analysis
├── experiment_tipping_points.py  # Degenerate strategy experiment
├── process_results.py            # Post-processing of simulation logs
├── stats_tracker.py              # Per-step statistics collection
├── log_simulation.py             # JSONL logging helpers
├── make_sensitivity_table.py     # Generate LaTeX Sobol sensitivity table
├── visualizations_*.ipynb        # Jupyter notebooks for figures
└── results/                      # Pre-computed results for reproducing figures
```

## Run the Simulation (~30')

The default entry point runs **parallel simulations** across 10 random seeds for three reward types (`reputation`, `raw_pubcount`, `h_index`) using the calibrated parameters and the `multiply` reward distribution mode.

```bash
python run_policy_simulation.py
```

This launches up to 10 parallel workers per reward type. Each run creates the following output files in `log/`:

| File | Contents |
|------|----------|
| `log/balanced_<reward_type>_multiply_seed<N>_actions.jsonl` | Agent policy assignments per step |
| `log/balanced_<reward_type>_multiply_seed<N>_observations.jsonl` | Observations per step |
| `log/balanced_<reward_type>_multiply_seed<N>_projects.json` | Final project states |

Additionally, a `careerist_vs_random` scenario is run for the combined (`all`) reward type.

### Customising a Run

Edit the `if __name__ == "__main__"` block in `run_policy_simulation.py`, or call `run_simulation_with_policies(...)` directly. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 3000 | Maximum agent pool size |
| `start_agents` | 200 | Agents active at step 0 |
| `max_steps` | 600 | Simulation length |
| `n_groups` | 20 | Number of peer groups |
| `max_peer_group_size` | 150 | Max agents per peer group |
| `policy_distribution` | equal thirds | Dict mapping policy names to proportions |
| `reward_type` | `"all"` | `"reputation"`, `"h_index"`, `"raw_pubcount"`, or `"all"` |
| `distribution_mode` | `"multiply"` | How rewards are split: `"multiply"`, `"evenly"`, or `"by_effort"` |
| `seed` | 42 | Random seed |

Predefined policy mixes are available in `POLICY_CONFIGS` at the top of the file.

## Run Sensitivity (~12h) and Calibration (~12h)

Calibration fits simulation parameters to real-world data distributions (papers per author, authors per paper, career lifespan, quality, acceptance rate). **This requires access to the real-world datasets** (`.npy` files in the repo root).

```bash
python calibrate.py
```

The script performs multi-seed Bayesian optimisation (5 seeds per evaluation) with a variance penalty to find robust parameter settings. Results are printed as a `CALIBRATED_PARAMS` dict that can be pasted into `run_policy_simulation.py`.

## Run the Tipping-Point Experiment (~30')

The tipping-point experiment sweeps the proportion of **adverse agents** from 0% to 100% (in 5% increments) and measures how societal outcomes degrade. It runs for three reward types (`reputation`, `raw_pubcount`, `h_index`), using the calibrated parameters.

```bash
python experiment_tipping_points.py
```

- Launches **21 simulations per reward type** (63 total) in parallel using `ProcessPoolExecutor`.
- Each simulation uses 3000 agents, 600 steps, 20 homogeneous peer groups.
- The adverse agent proportion replaces equal shares of the three regular archetypes.

### Output

Results are saved to `log/`:

| File | Contents |
|------|----------|
| `log/tipping_point_experiment_reputation_results.json` | Per-proportion summary for reputation reward |
| `log/tipping_point_experiment_raw_pubcount_results.json` | Per-proportion summary for pubcount reward |
| `log/tipping_point_experiment_h_index_results.json` | Per-proportion summary for h-index reward |

Each JSON file contains an array of objects with:
- `adverse_proportion` — fraction of adverse agents (0.0–1.0)
- `total_societal_value`, `total_rewards_distributed`, `successful_projects_count`
- `total_terminations`, `avg_agent_age`, `success_rate`
- `final_populations` — surviving population by archetype

Visualise results with `visualize_tipping_points.ipynb`.

## Agent Policies (in `agent_policies.py`)

- **Careerist** (`careerist`)
  - Picks high-prestige opportunities above a threshold.
  - Collaborates with active peers at or above the active-peer average reputation.
  - Effort goes to the closest-deadline running project that still needs work.

- **Orthodox Scientist** (`orthodox_scientist`)
  - Prefers lowest-novelty opportunities; ties break toward higher prestige.
  - Collaborates with all active peers who have close topic centroids.
  - Effort prioritizes projects already below 90% of required effort; otherwise best peer fit.

- **Mass Producer** (`mass_producer`)
  - Takes a project if the (effort × time window) is relatively low.
  - Collaborates with all active peers within the action mask.
  - Effort goes to the closest-deadline running project that still needs work.

**Shared safety net:** if a running project risks missing its requirement given remaining time, policies skip selecting a new project that step. Additionally, agents whose current mean workload ratio ≥ 1.0 are blocked from starting new projects.

## Post-Processing & Visualisation

### Process Results

Aggregate simulation logs into per-archetype reward summaries (requires completed simulation runs for all 10 seeds) or reproduce figures from pre-computed results in `results/`:

- `visualizations_summary_plots.ipynb` — summary statistics and reward distributions
- `visualizations_reward_type.ipynb` — reward-type comparison plots
- `visualize_tipping_points.ipynb` — tipping-point experiment plots

### Sensitivity Analysis Table

Generate the LaTeX Sobol sensitivity table (requires sensitivity analysis JSON files):

```bash
python make_sensitivity_table.py
```

## Tips
- Simulations are CPU-intensive; a machine with ≥ 10 cores is recommended for the full parallel runs.
- JSONL logs can be large; use `jq`, `tail -f`, or sample lines.
- Commit parameter changes alongside their summary files for reproducibility.
- to reproduce figure from saved files make sure not to run the second cell in the first two notebooks
