#!/usr/bin/env python3
"""
Plot publication-quality bar charts from genome re-evaluation JSON output.

Reads JSON produced by eval/evaluate_genomes.py and creates:
  1. Grouped bar plot of mean reward per task per algorithm
  2. Normalized version (divided by reference rewards)

Usage:
    python plot_evaluation_results.py --input ../../eval/outputs/multi_task_neat_vs_haneat_ablation_*.json
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants (must match plot_training_curves.py)
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

REFERENCE_REWARDS = {
    "hopper": 3000,
    "walker2d": 5000,
    "ant": 6000,
    "halfcheetah": 8000,
    "humanoid": 6000,
    "reacher": 5,
}

ALGO_DOT_COLORS = {
    "neat": "#164F7A",
    "ha_neat": "#9E3526",
    "ha_neat_ablation": "#9E3526",
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
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(input_path: str) -> dict:
    with open(input_path) as f:
        return json.load(f)


def detect_algorithm(run: dict) -> str:
    """Detect algorithm from filename (more precise than the 'algorithm' field)."""
    fname = run.get("file", "")
    if fname.startswith("ha_neat_ablation"):
        return "ha_neat_ablation"
    elif fname.startswith("ha_neat"):
        return "ha_neat"
    return "neat"


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_evaluation(data: dict, tasks: list[str], output_dir: str, fmt: str, normalized: bool = False):
    """Grouped bar plot: one group per task, one bar per algorithm."""
    runs = [r for r in data["runs"] if "error" not in r]

    # Group all episodes by algorithm and task
    algo_task_episodes: dict[str, dict[str, list[float]]] = {}
    algo_task_seed_means: dict[str, dict[str, list[float]]] = {}

    for run in runs:
        algo = detect_algorithm(run)
        if algo not in algo_task_episodes:
            algo_task_episodes[algo] = {t: [] for t in tasks}
            algo_task_seed_means[algo] = {t: [] for t in tasks}

        for task in tasks:
            if task in run.get("tasks", {}):
                episodes = run["tasks"][task]["episodes"]
                algo_task_episodes[algo][task].extend(episodes)
                algo_task_seed_means[algo][task].append(run["tasks"][task]["mean"])

    algos = sorted(algo_task_episodes.keys())
    n_algos = len(algos)
    n_tasks = len(tasks)

    bar_width = 0.3
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, algo in enumerate(algos):
        color = ALGO_COLORS.get(algo, DEFAULT_COLOR)
        dot_color = ALGO_DOT_COLORS.get(algo, "#333333")
        label = ALGO_DISPLAY.get(algo, algo)

        means = []
        sems = []
        seed_means_per_task = []

        for task in tasks:
            episodes = np.array(algo_task_episodes[algo][task])
            ref = REFERENCE_REWARDS.get(task, 1.0) if normalized else 1.0
            eps = episodes / ref

            means.append(np.mean(eps))
            sems.append(np.std(eps) / np.sqrt(len(eps)))
            seed_means_per_task.append(np.array(algo_task_seed_means[algo][task]) / ref)

        offset = (i - (n_algos - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            color=color, edgecolor=color, alpha=0.85,
            label=label, yerr=sems, capsize=3,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
        )

        # Overlay individual seed means as dots
        for j, seed_vals in enumerate(seed_means_per_task):
            jitter = np.random.default_rng(42).uniform(-bar_width * 0.2, bar_width * 0.2, len(seed_vals))
            ax.scatter(
                np.full_like(seed_vals, x[j] + offset) + jitter,
                seed_vals,
                color=dot_color, s=20, zorder=5, edgecolors="white", linewidths=0.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_ylabel("Normalized fitness" if normalized else "Mean reward")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.yaxis.grid(True, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)

    experiment_name = data["metadata"]["experiment"]
    suffix = "_normalized" if normalized else ""
    out_path = os.path.join(output_dir, f"{experiment_name}_evaluation{suffix}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results from JSON.")
    parser.add_argument("--input", required=True, help="Path to evaluation JSON file")
    parser.add_argument("--output_dir", default="./", help="Output directory")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], dest="fmt")
    parser.add_argument("--tasks", default="hopper,walker2d", help="Comma-separated task names")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data = load_results(args.input)

    # Print summary
    runs = [r for r in data["runs"] if "error" not in r]
    algos = set(detect_algorithm(r) for r in runs)
    print(f"Loaded {len(runs)} runs, {len(algos)} algorithm(s): {sorted(algos)}")
    print(f"Episodes per genome: {data['metadata']['n_eval']}")

    for algo in sorted(algos):
        algo_runs = [r for r in runs if detect_algorithm(r) == algo]
        print(f"\n  {ALGO_DISPLAY.get(algo, algo)} ({len(algo_runs)} seeds):")
        for task in tasks:
            all_eps = []
            for r in algo_runs:
                if task in r.get("tasks", {}):
                    all_eps.extend(r["tasks"][task]["episodes"])
            if all_eps:
                arr = np.array(all_eps)
                ref = REFERENCE_REWARDS.get(task, 1.0)
                print(f"    {task:10s}  mean={np.mean(arr):.1f}  std={np.std(arr):.1f}  "
                      f"norm={np.mean(arr)/ref:.3f}")

    print()
    setup_matplotlib()
    plot_evaluation(data, tasks, output_dir, args.fmt, normalized=False)
    plot_evaluation(data, tasks, output_dir, args.fmt, normalized=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
