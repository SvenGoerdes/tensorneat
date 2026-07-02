#!/usr/bin/env python3
"""Generational dynamics analysis for the pop200 experiment.

Reads per-generation MLflow metrics for experiment
``multi_task_neat_vs_haneat_pop200`` (30 seeds x 3 conditions: neat, ha_neat,
ha_neat_ablation) and produces:

  1. complexity_over_generations.png  -- best-genome node & connection counts
     over generations (mean +/- IQR band) per condition.
  2. species_over_generations.png     -- species count over generations
     (mean +/- IQR band) per condition.

Plus a Kruskal-Wallis test on the per-seed mean species count over the final
100 generations, and summary stats written to generational_dynamics.md.

Metrics used (logged in pipeline.py per generation):
  neat/best_genome_n_nodes, neat/best_genome_n_conns, neat/n_species

Read-only access to mlflow.db. Reproducible: run from repo root with

    uv run python analysis/generational_dynamics_pop200.py
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "mlflow.db")
EXPERIMENT_NAME = "multi_task_neat_vs_haneat_pop200"
OUTPUT_DIR = os.path.join(REPO_ROOT, "analysis", "outputs", "pop200")

CONDITIONS = ["neat", "ha_neat", "ha_neat_ablation"]

ALGO_COLORS = {
    "neat": "#2176AE",
    "ha_neat": "#D64933",
    "ha_neat_ablation": "#4C9A2A",
}
ALGO_DISPLAY = {
    "neat": "NEAT",
    "ha_neat": "HA-NEAT",
    "ha_neat_ablation": "HA-NEAT (tanh only)",
}

METRIC_NODES = "neat/best_genome_n_nodes"
METRIC_CONNS = "neat/best_genome_n_conns"
METRIC_SPECIES = "neat/n_species"
METRICS = [METRIC_NODES, METRIC_CONNS, METRIC_SPECIES]

FINAL_WINDOW = 100  # generations for final-window summary stats


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#cccccc",
        "lines.linewidth": 1.5,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def condition_of(run_name: str) -> str:
    for prefix in ("ha_neat_ablation", "ha_neat", "neat"):
        if run_name.startswith(prefix + "_"):
            return prefix
    return run_name.split("_normalized")[0]


def load_experiment(db_path: str, experiment_name: str, metric_keys: list[str]):
    """Load per-generation metrics.

    Returns {condition: {metric_key: 2D array [n_seeds, n_steps]}, "_steps": arr}.
    Deduplicates by keeping the run (per distinct name) with the most metric rows,
    matching the convention in text/figures/plot_training_curves.py.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    exp = cur.execute(
        "SELECT experiment_id FROM experiments WHERE name = ?", (experiment_name,)
    ).fetchone()
    if exp is None:
        raise SystemExit(f"Experiment '{experiment_name}' not found.")
    experiment_id = exp[0]

    runs = cur.execute(
        "SELECT run_uuid, name FROM runs WHERE experiment_id = ?", (experiment_id,)
    ).fetchall()

    # Deduplicate by name: keep the uuid with the most metric rows.
    name_to_uuids: dict[str, list[str]] = defaultdict(list)
    for run_uuid, name in runs:
        name_to_uuids[name].append(run_uuid)

    name_to_uuid: dict[str, str] = {}
    for name, uuids in name_to_uuids.items():
        if len(uuids) == 1:
            name_to_uuid[name] = uuids[0]
            continue
        best_uuid, best_count = None, -1
        for uuid in uuids:
            count = cur.execute(
                "SELECT COUNT(*) FROM metrics WHERE run_uuid = ?", (uuid,)
            ).fetchone()[0]
            if count > best_count:
                best_uuid, best_count = uuid, count
        name_to_uuid[name] = best_uuid

    # Collect per-condition series keyed by (metric, step).
    placeholders = ",".join("?" for _ in metric_keys)
    # {condition: {metric: {run_name: {step: value}}}}
    raw: dict[str, dict[str, dict[str, dict[int, float]]]] = {
        c: {m: {} for m in metric_keys} for c in CONDITIONS
    }

    for name, uuid in name_to_uuid.items():
        cond = condition_of(name)
        if cond not in raw:
            continue
        rows = cur.execute(
            f"SELECT key, step, value FROM metrics "
            f"WHERE run_uuid = ? AND key IN ({placeholders})",
            (uuid, *metric_keys),
        ).fetchall()
        for key, step, value in rows:
            raw[cond][key].setdefault(name, {})[int(step)] = value

    conn.close()

    # Determine common step grid (intersection of steps present across all runs).
    all_steps: set[int] = set()
    for cond in CONDITIONS:
        for name, sv in raw[cond][METRIC_SPECIES].items():
            all_steps |= set(sv.keys())
    steps = np.array(sorted(all_steps), dtype=int)

    result: dict[str, dict[str, np.ndarray]] = {}
    for cond in CONDITIONS:
        result[cond] = {}
        for metric in metric_keys:
            run_names = sorted(raw[cond][metric].keys())
            mat = np.full((len(run_names), len(steps)), np.nan)
            for i, name in enumerate(run_names):
                sv = raw[cond][metric][name]
                for j, s in enumerate(steps):
                    if s in sv:
                        mat[i, j] = sv[s]
            result[cond][metric] = mat
    result["_steps"] = steps
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_panel(ax, steps, data, metric, ylabel):
    for cond in CONDITIONS:
        mat = data[cond][metric]
        if mat.size == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        color = ALGO_COLORS[cond]
        ax.plot(steps, mean, color=color, linewidth=2.0, label=ALGO_DISPLAY[cond])
        ax.fill_between(steps, q25, q75, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_xlim(steps.min(), steps.max())


def plot_complexity(steps, data, out_path):
    fig, (ax_nodes, ax_conns) = plt.subplots(1, 2, figsize=(11, 4.3))
    plot_metric_panel(ax_nodes, steps, data, METRIC_NODES,
                      "Best-genome node count")
    plot_metric_panel(ax_conns, steps, data, METRIC_CONNS,
                      "Best-genome connection count")
    ax_nodes.set_title("(a) Nodes")
    ax_conns.set_title("(b) Connections")
    ax_nodes.legend(framealpha=0.9, edgecolor="none")
    fig.suptitle("Network complexity over generations (mean, shaded IQR)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_species(steps, data, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_metric_panel(ax, steps, data, METRIC_SPECIES, "Species count")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_title("Species count over generations (mean, shaded IQR)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def final_window_means(steps, data, metric, window):
    """Per-seed mean of `metric` over the final `window` generations."""
    cutoff = steps.max() - window + 1
    mask = steps >= cutoff
    out = {}
    for cond in CONDITIONS:
        mat = data[cond][metric][:, mask]
        out[cond] = np.nanmean(mat, axis=1)  # one value per seed
    return out, int(cutoff)


def summarize(values):
    v = np.asarray(values)
    return dict(n=int(v.size), mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path, steps, data, species_final, species_cutoff,
                 kw_stat, kw_p, nodes_final, conns_final):
    lines = []
    lines.append("# Generational Dynamics — pop200 experiment\n")
    lines.append(f"Experiment: `{EXPERIMENT_NAME}` (MLflow experiment_id 14)\n")
    lines.append(
        f"Population 200, {steps.max()} generations, "
        f"{data['neat'][METRIC_SPECIES].shape[0]} NEAT / "
        f"{data['ha_neat'][METRIC_SPECIES].shape[0]} HA-NEAT / "
        f"{data['ha_neat_ablation'][METRIC_SPECIES].shape[0]} ablation seeds.\n"
    )
    lines.append(
        "Runs deduplicated by name (kept the run with the most logged steps). "
        "Bands in the figures are the inter-quartile range (IQR, 25th-75th "
        "percentile) across seeds; lines are the across-seed mean.\n"
    )

    lines.append("## Metrics available in MLflow (per generation)\n")
    lines.append(
        "- `neat/best_genome_n_nodes` — node count of the best genome\n"
        "- `neat/best_genome_n_conns` — connection count of the best genome\n"
        "- `neat/avg_genome_n_conns` — population-mean connection count\n"
        "- `neat/n_species` — number of non-empty species\n"
    )

    def table(title, final_dict):
        rows = [f"### {title}\n",
                "| Condition | n | mean | std | median | IQR |",
                "|---|---|---|---|---|---|"]
        for cond in CONDITIONS:
            s = summarize(final_dict[cond])
            rows.append(
                f"| {ALGO_DISPLAY[cond]} | {s['n']} | {s['mean']:.2f} | "
                f"{s['std']:.2f} | {s['median']:.2f} | "
                f"[{s['q25']:.2f}, {s['q75']:.2f}] |"
            )
        return "\n".join(rows) + "\n"

    lines.append(
        f"## Final-window summary (generations {species_cutoff}–{steps.max()}, "
        f"last {FINAL_WINDOW})\n"
    )
    lines.append(table("Species count (per-seed mean over final window)", species_final))
    lines.append(table("Best-genome node count (per-seed mean over final window)", nodes_final))
    lines.append(table("Best-genome connection count (per-seed mean over final window)", conns_final))

    lines.append("## Kruskal–Wallis test — final-window species count\n")
    lines.append(
        f"Across the three conditions on the per-seed mean species count over the "
        f"final {FINAL_WINDOW} generations:\n\n"
        f"- H = {kw_stat:.3f}, p = {kw_p:.4f}\n"
    )
    verdict = "a significant" if kw_p < 0.05 else "no significant"
    lines.append(
        f"There is {verdict} difference in sustained species count across "
        f"conditions (alpha = 0.05).\n"
    )

    # Interpretation
    node_means = {c: np.mean(nodes_final[c]) for c in CONDITIONS}
    conn_means = {c: np.mean(conns_final[c]) for c in CONDITIONS}
    sp_means = {c: np.mean(species_final[c]) for c in CONDITIONS}
    lines.append("## Interpretation\n")
    lines.append(
        f"NEAT sustains a mean best-genome node count of {node_means['neat']:.2f} in the "
        f"final window versus {node_means['ha_neat']:.2f} (HA-NEAT) and "
        f"{node_means['ha_neat_ablation']:.2f} (ablation), and a connection count of "
        f"{conn_means['neat']:.2f} versus {conn_means['ha_neat']:.2f} / "
        f"{conn_means['ha_neat_ablation']:.2f} — consistent with the final-genome finding "
        f"that NEAT accretes more structure at equal fitness. The complexity trajectories "
        f"show when this gap opens up (see complexity_over_generations.png). "
    )
    if kw_p < 0.05:
        lines.append(
            f"Species counts differ significantly across conditions (mean "
            f"{sp_means['neat']:.2f}/{sp_means['ha_neat']:.2f}/"
            f"{sp_means['ha_neat_ablation']:.2f} for NEAT/HA-NEAT/ablation), suggesting "
            f"HA-NEAT's speciation protection does reshape the sustained species pool.\n"
        )
    else:
        lines.append(
            f"Species counts are statistically indistinguishable across conditions (mean "
            f"{sp_means['neat']:.2f}/{sp_means['ha_neat']:.2f}/"
            f"{sp_means['ha_neat_ablation']:.2f} for NEAT/HA-NEAT/ablation), so HA-NEAT's "
            f"speciation protection does not translate into a measurably larger sustained "
            f"species pool at this population size.\n"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Database:   {DB_PATH}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    data = load_experiment(DB_PATH, EXPERIMENT_NAME, METRICS)
    steps = data["_steps"]
    print(f"Steps: {steps.min()}–{steps.max()} ({len(steps)} points)")
    for cond in CONDITIONS:
        print(f"  {ALGO_DISPLAY[cond]:22s} {data[cond][METRIC_SPECIES].shape[0]} seeds")

    setup_matplotlib()
    plot_complexity(steps, data, os.path.join(OUTPUT_DIR, "complexity_over_generations.png"))
    plot_species(steps, data, os.path.join(OUTPUT_DIR, "species_over_generations.png"))

    species_final, cutoff = final_window_means(steps, data, METRIC_SPECIES, FINAL_WINDOW)
    nodes_final, _ = final_window_means(steps, data, METRIC_NODES, FINAL_WINDOW)
    conns_final, _ = final_window_means(steps, data, METRIC_CONNS, FINAL_WINDOW)

    kw_stat, kw_p = stats.kruskal(*[species_final[c] for c in CONDITIONS])
    print(f"Kruskal-Wallis (species, final {FINAL_WINDOW} gens): H={kw_stat:.3f} p={kw_p:.4f}")

    write_report(
        os.path.join(OUTPUT_DIR, "generational_dynamics.md"),
        steps, data, species_final, cutoff, kw_stat, kw_p, nodes_final, conns_final,
    )
    print("Done.")


if __name__ == "__main__":
    main()
