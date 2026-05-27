import json
from pathlib import Path

# Mapping from JSON filenames to table column labels
METRICS = {
    "sensitivity_papers_per_author.json": "PPA",
    "sensitivity_authors_per_paper.json": "APP",
    "sensitivity_lifespan.json": "CL",
    "sensitivity_quality.json": "PQ",
    "sensitivity_acceptance.json": "AR",
}

# Mapping from parameter keys in JSON to table labels
PARAMETERS = {
    "acceptance_threshold": "Acceptance threshold",
    "orthodox_novelty_threshold": "Novelty threshold",
    "careerist_prestige_threshold": "Prestige threshold",
    "mass_producer_effort_threshold": "Effort threshold",
    "coordination_factor": "Coordination factor",
    "max_rewardless_steps": "Max rewardless steps",
    "continuation_probability": "Continuation probability",
}

# Threshold for bold formatting
BOLD_THRESHOLD = 0.10

# Directory containing JSON files
DATA_DIR = Path(".")

# Store parsed results
results = {}

# Load all metric files
for filename, metric_label in METRICS.items():
    filepath = DATA_DIR / filename

    with open(filepath, "r") as f:
        data = json.load(f)

    results[metric_label] = data

# Build LaTeX table rows
table_rows = []

for param_key, param_label in PARAMETERS.items():

    row = [param_label]

    for filename, metric_label in METRICS.items():

        s1 = results[metric_label]["S1"].get(param_key, 0.0)
        st = results[metric_label]["ST"].get(param_key, 0.0)

        # Optional: treat tiny negative Sobol estimates as zero
        if s1 < 0 and abs(s1) < 0.05:
            s1 = 0.0
        if st < 0 and abs(st) < 0.05:
            st = 0.0

        s1_fmt = f"{s1:.2f}"
        st_fmt = f"{st:.2f}"

        # Bold values above threshold
        if abs(s1) >= BOLD_THRESHOLD:
            s1_fmt = f"\\textbf{{{s1_fmt}}}"

        if abs(st) >= BOLD_THRESHOLD:
            st_fmt = f"\\textbf{{{st_fmt}}}"

        row.append(f"({s1_fmt}, {st_fmt})")

    table_rows.append(" & ".join(row) + r" \\")

# Generate complete LaTeX table
latex_table = r"""
\begin{table}[t]
\centering
\caption{Sobol sensitivity indices ($S_j$ first-order, $S^T_j$ total-order) for key parameters' influence on variance of model output metrics.}
\label{tab:sensitivity}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Parameter} & \textbf{PPA} & \textbf{APP} & \textbf{CL} & \textbf{PQ} & \textbf{AR} \\
 & $(S_j, S^{T}_j)$ & $(S_j, S^{T}_j)$ & $(S_j, S^{T}_j)$ & $(S_j, S^{T}_j)$ & $(S_j, S^{T}_j)$ \\
\midrule
"""

latex_table += "\n".join(table_rows)

latex_table += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""

# Save output
output_file = "sobol_sensitivity_table.tex"

with open(output_file, "w") as f:
    f.write(latex_table)

print(f"LaTeX table written to: {output_file}")
print()
print(latex_table)