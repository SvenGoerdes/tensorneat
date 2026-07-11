#!/usr/bin/env python3
"""Training-curve figure for the pop200 experiment (thesis Results chapter).

Reads the per-generation population-best fitness (``fitness/max``) for
experiment ``multi_task_neat_vs_haneat_pop200`` (30 seeds x 3 conditions) and
produces:

  fitness_over_generations.png -- per-generation best normalized-minimum
      fitness, across-seed mean with a 95% confidence band per condition.

This is the "learning dynamics" measurement level promised in the thesis's
Evaluation Protocol section (fitness-over-generation curves, averaged across
runs with 95% confidence bands).

Read-only access to mlflow.db. Reproducible: run from repo root with

    uv run python analysis/training_curves_pop200.py
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

METRIC_FITNESS = "fitness/max"

CONFIDENCE = 0.95


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


def load_fitness(db_path: str, experiment_name: str):
    """Load per-generation fitness/max.

    Returns ({condition: 2D array [n_seeds, n_steps]}, steps).
    Deduplicates by keeping the run (per distinct name) with the most metric
    rows, matching analysis/generational_dynamics_pop200.py.
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
        "SELECT run_uuid, name FROM runs WHERE experiment_id = ? AND status = 'FINISHED'",
        (experiment_id,),
    ).fetchall()

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

    raw: dict[str, dict[str, dict[int, float]]] = {c: {} for c in CONDITIONS}
    for name, uuid in name_to_uuid.items():
        cond = condition_of(name)
        if cond not in raw:
            continue
        rows = cur.execute(
            "SELECT step, value FROM metrics WHERE run_uuid = ? AND key = ?",
            (uuid, METRIC_FITNESS),
        ).fetchall()
        raw[cond][name] = {int(step): value for step, value in rows}

    conn.close()

    all_steps: set[int] = set()
    for cond in CONDITIONS:
        for sv in raw[cond].values():
            all_steps |= set(sv.keys())
    steps = np.array(sorted(all_steps), dtype=int)

    result: dict[str, np.ndarray] = {}
    for cond in CONDITIONS:
        run_names = sorted(raw[cond].keys())
        mat = np.full((len(run_names), len(steps)), np.nan)
        for i, name in enumerate(run_names):
            sv = raw[cond][name]
            for j, s in enumerate(steps):
                if s in sv:
                    mat[i, j] = sv[s]
        result[cond] = mat
    return result, steps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fitness_curves(data: dict[str, np.ndarray], steps: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cond in CONDITIONS:
        mat = data[cond]
        if mat.size == 0:
            continue
        n = np.sum(~np.isnan(mat), axis=0)
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1)
        color = ALGO_COLORS[cond]
        ax.plot(steps, mean, color=color, linewidth=2.0, label=ALGO_DISPLAY[cond])
        ax.fill_between(
            steps, mean - t_crit * sem, mean + t_crit * sem,
            color=color, alpha=0.18, linewidth=0,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best normalized-minimum fitness")
    ax.legend(framealpha=0.9, edgecolor="none", loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data, steps = load_fitness(DB_PATH, EXPERIMENT_NAME)

    for cond in CONDITIONS:
        print(f"{ALGO_DISPLAY[cond]:22s} {data[cond].shape[0]} seeds, "
              f"{data[cond].shape[1]} generations")

    setup_matplotlib()
    out_path = os.path.join(OUTPUT_DIR, "fitness_over_generations.png")
    plot_fitness_curves(data, steps, out_path)


if __name__ == "__main__":
    main()
