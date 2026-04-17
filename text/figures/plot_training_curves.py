#!/usr/bin/env python3
"""
Plot publication-quality training curves from MLflow SQLite database.

Reads per-generation metrics from an MLflow experiment and produces:
  1. Per-task training curves (one figure per task)
  2. Aggregated fitness curve (fitness/max)

Usage:
    python plot_training_curves.py --experiment_name multi_task_neat_vs_haneat_ablation
    python plot_training_curves.py --experiment_name my_exp --tasks hopper,walker2d --format png
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    "neat": "#2176AE",
    "ha_neat": "#D64933",
    "ha_neat_ablation": "#D64933",
}

ALGO_DISPLAY = {
    "neat": "NEAT",
    "ha_neat": "HA-NEAT",
    "ha_neat_ablation": "HA-NEAT (tanh only)",
}

DEFAULT_COLOR = "#555555"


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def setup_matplotlib():
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

def parse_algorithm(run_name: str) -> str:
    """Extract algorithm type from run name (everything before '_normalized')."""
    parts = run_name.split("_normalized")
    if len(parts) >= 2:
        return parts[0]
    for prefix in ("ha_neat_ablation", "ha_neat", "neat"):
        if run_name.startswith(prefix + "_"):
            return prefix
    return run_name


def load_metrics(
    db_path: str,
    experiment_name: str,
    metric_keys: list[str],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load metrics from the MLflow SQLite database.

    Returns {run_name: {metric_key: [(step, value), ...]}}.
    Only FINISHED runs. Deduplicates by keeping the run with most steps.
    """
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT experiment_id FROM experiments WHERE name = ?", (experiment_name,))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT name FROM experiments")
        available = [r[0] for r in cur.fetchall()]
        print(f"ERROR: Experiment '{experiment_name}' not found.\nAvailable: {available}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    experiment_id = row[0]

    cur.execute(
        "SELECT run_uuid, name FROM runs WHERE experiment_id = ? AND status = 'FINISHED'",
        (experiment_id,),
    )
    runs_raw = cur.fetchall()
    if not runs_raw:
        print(f"ERROR: No FINISHED runs for '{experiment_name}'.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    # Deduplicate: keep run with most metric rows per name
    name_to_candidates: dict[str, list[str]] = defaultdict(list)
    for run_uuid, name in runs_raw:
        name_to_candidates[name].append(run_uuid)

    name_to_uuid: dict[str, str] = {}
    for name, uuids in name_to_candidates.items():
        if len(uuids) == 1:
            name_to_uuid[name] = uuids[0]
        else:
            best_uuid, best_count = None, -1
            for uuid in uuids:
                cur.execute("SELECT COUNT(*) FROM metrics WHERE run_uuid = ?", (uuid,))
                count = cur.fetchone()[0]
                if count > best_count:
                    best_uuid, best_count = uuid, count
            name_to_uuid[name] = best_uuid

    # Load metrics
    placeholders = ",".join("?" for _ in metric_keys)
    result: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for run_name, run_uuid in name_to_uuid.items():
        cur.execute(
            f"SELECT key, step, value FROM metrics "
            f"WHERE run_uuid = ? AND key IN ({placeholders}) ORDER BY step",
            (run_uuid, *metric_keys),
        )
        metrics_dict: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for key, step, value in cur.fetchall():
            metrics_dict[key].append((step, value))
        if metrics_dict:
            result[run_name] = dict(metrics_dict)

    conn.close()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _algo_color(algo: str) -> str:
    return ALGO_COLORS.get(algo, DEFAULT_COLOR)


def _algo_label(algo: str) -> str:
    return ALGO_DISPLAY.get(algo, algo)


def _group_by_algorithm(data):
    groups: dict[str, list[str]] = defaultdict(list)
    for run_name in data:
        groups[parse_algorithm(run_name)].append(run_name)
    return dict(groups)


def _extract_series(steps_values):
    if not steps_values:
        return np.array([]), np.array([])
    arr = np.array(steps_values)
    order = np.argsort(arr[:, 0])
    return arr[order, 0].astype(int), arr[order, 1]


def _compute_mean_curve(all_steps, all_values):
    if not all_steps:
        return np.array([]), np.array([])
    common_steps = np.unique(np.concatenate(all_steps))
    interpolated = np.full((len(all_steps), len(common_steps)), np.nan)
    for i, (steps, values) in enumerate(zip(all_steps, all_values)):
        if len(steps) == 0:
            continue
        interpolated[i] = np.interp(common_steps, steps, values)
    return common_steps, np.nanmean(interpolated, axis=0)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_per_task(data, task, metric_prefix, experiment_name, output_dir, fmt):
    metric_key = f"{metric_prefix}/{task}"
    algo_groups = _group_by_algorithm(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for algo in sorted(algo_groups.keys()):
        color = _algo_color(algo)
        label = _algo_label(algo)
        all_steps, all_values = [], []

        for rn in algo_groups[algo]:
            if metric_key not in data[rn]:
                continue
            steps, values = _extract_series(data[rn][metric_key])
            if len(steps) == 0:
                continue
            all_steps.append(steps)
            all_values.append(values)
            ax.plot(steps, values, color=color, alpha=0.35, linewidth=0.9, linestyle="--")

        if all_steps:
            mean_steps, mean_vals = _compute_mean_curve(all_steps, all_values)
            ax.plot(mean_steps, mean_vals, color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Generation")
    task_display = task.capitalize()
    if "normalized" in metric_prefix:
        ax.set_ylabel(f"Best normalized fitness ({task_display})")
    else:
        ax.set_ylabel(f"Best fitness ({task_display})")

    ax.legend(framealpha=0.9, edgecolor="none")
    fig.tight_layout(pad=0.4)

    out_path = os.path.join(output_dir, f"{experiment_name}_training_{task}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_aggregated(data, experiment_name, output_dir, fmt):
    metric_key = "fitness/max"
    algo_groups = _group_by_algorithm(data)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for algo in sorted(algo_groups.keys()):
        color = _algo_color(algo)
        label = _algo_label(algo)
        all_steps, all_values = [], []

        for rn in algo_groups[algo]:
            if metric_key not in data[rn]:
                continue
            steps, values = _extract_series(data[rn][metric_key])
            if len(steps) == 0:
                continue
            all_steps.append(steps)
            all_values.append(values)
            ax.plot(steps, values, color=color, alpha=0.35, linewidth=0.9, linestyle="--")

        if all_steps:
            mean_steps, mean_vals = _compute_mean_curve(all_steps, all_values)
            ax.plot(mean_steps, mean_vals, color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best aggregated fitness")
    ax.legend(framealpha=0.9, edgecolor="none")
    fig.tight_layout(pad=0.4)

    out_path = os.path.join(output_dir, f"{experiment_name}_training_aggregated.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot training curves from MLflow SQLite database.")
    parser.add_argument("--db_path", default="../../mlflow.db", help="Path to MLflow SQLite DB")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name")
    parser.add_argument("--output_dir", default="./", help="Output directory")
    parser.add_argument("--metric_prefix", default="best_fitness_normalized", help="Metric prefix for per-task plots")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], dest="fmt")
    parser.add_argument("--tasks", default="hopper,walker2d", help="Comma-separated task names")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isabs(args.db_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.normpath(os.path.join(script_dir, args.db_path))
    else:
        db_path = args.db_path

    metric_keys = [f"{args.metric_prefix}/{t}" for t in tasks] + ["fitness/max"]

    print(f"Database:   {db_path}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Tasks:      {tasks}")
    print(f"Metrics:    {metric_keys}")
    print()

    data = load_metrics(db_path, args.experiment_name, metric_keys)

    algo_groups = _group_by_algorithm(data)
    print(f"Loaded {len(data)} runs across {len(algo_groups)} algorithm(s):")
    for algo in sorted(algo_groups.keys()):
        run_names = algo_groups[algo]
        max_steps = max(len(data[rn][mk]) for rn in run_names for mk in data[rn])
        print(f"  {_algo_label(algo):25s}  {len(run_names)} run(s), up to {max_steps} steps")
    print()

    setup_matplotlib()
    for task in tasks:
        plot_per_task(data, task, args.metric_prefix, args.experiment_name, output_dir, args.fmt)
    plot_aggregated(data, args.experiment_name, output_dir, args.fmt)
    print("\nDone.")


if __name__ == "__main__":
    main()
