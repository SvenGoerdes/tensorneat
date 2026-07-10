#!/usr/bin/env python3
"""Training-budget calibration figure.

Justifies the 750-generation training budget of the main study by showing that
best-so-far aggregated fitness plateaus well before generation 750 in the long
calibration runs (population 1024-4096, up to 2000 generations) and that the
same convergence-by-750 behaviour holds at the main-study population of 200.

The plotted quantity is the running (cumulative) maximum of the per-generation
aggregated fitness `fitness/max`. Because NEAT is elitist, the population max
each generation equals the best-so-far genome fitness; its final value matches
the MLflow `best_fitness` summary metric exactly (verified). Taking the
cumulative max yields a clean, monotone best-so-far convergence curve.

Metric mapping (all experiments use the same keys):
  fitness/max  -> per-generation best aggregated fitness (normalized_min)
  step         -> generation index (1..N)

Long calibration runs (never named by internal codename in thesis text):
  multi_task_neat_vs_haneat_v4        pop 4096, 1000 gen  (NEAT + HA-NEAT)
  multi_task_neat_vs_haneat_final     pop 4096, 2000 gen  (NEAT + HA-NEAT)
  multi_task_haneat_finalv2           pop 4096, 2000 gen  (HA-NEAT)

Main study:
  multi_task_neat_vs_haneat_pop200    pop 200, 750 gen (NEAT, HA-NEAT, ablation)

Usage:
    uv run python text/figures/plot_budget_calibration.py
"""

import os
import sqlite3
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants (colours match plot_training_curves.py / plot_evaluation_results.py)
# ---------------------------------------------------------------------------

NEAT_COLOR = "#2176AE"
HANEAT_COLOR = "#D64933"
POP200_COLOR = "#3F8F4F"  # distinct green for the main-study overlay

BUDGET_GENERATION = 750
PLATEAU_START = 700  # generation from which "after" improvement is measured

# Long calibration runs: experiment name -> min generations required to keep a run
LONG_EXPERIMENTS = [
    "multi_task_neat_vs_haneat_v4",
    "multi_task_neat_vs_haneat_final",
    "multi_task_haneat_finalv2",
]
MAIN_EXPERIMENT = "multi_task_neat_vs_haneat_pop200"

METRIC_KEY = "fitness/max"


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

def _experiment_id(cur, name):
    cur.execute("SELECT experiment_id FROM experiments WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        print(f"ERROR: experiment '{name}' not found", file=sys.stderr)
        sys.exit(1)
    return row[0]


def _load_best_so_far(cur, experiment_name, name_filter=None):
    """Return {run_name: (gens, best_so_far)} for FINISHED runs.

    best_so_far is the cumulative max of fitness/max (monotone convergence
    curve). When multiple FINISHED runs share a name, keep the one with the
    most generations (dedup: the completed 750/1000/2000-gen run wins).
    """
    exp_id = _experiment_id(cur, experiment_name)
    cur.execute(
        "SELECT run_uuid, name FROM runs WHERE experiment_id = ? AND status = 'FINISHED'",
        (exp_id,),
    )
    candidates = defaultdict(list)
    for run_uuid, name in cur.fetchall():
        if name_filter is not None and not name_filter(name):
            continue
        candidates[name].append(run_uuid)

    out = {}
    for name, uuids in candidates.items():
        best_series = None
        for run_uuid in uuids:
            cur.execute(
                "SELECT step, value FROM metrics WHERE run_uuid = ? AND key = ? ORDER BY step",
                (run_uuid, METRIC_KEY),
            )
            rows = cur.fetchall()
            if not rows:
                continue
            arr = np.array(rows)
            order = np.argsort(arr[:, 0])
            gens = arr[order, 0].astype(int)
            vals = np.maximum.accumulate(arr[order, 1])  # best-so-far
            if best_series is None or len(gens) > len(best_series[0]):
                best_series = (gens, vals)
        if best_series is not None:
            out[name] = best_series
    return out


def _condition(name):
    if name.startswith("ha_neat_ablation"):
        return "ablation"
    if name.startswith("ha_neat"):
        return "ha_neat"
    return "neat"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean_band(series_list, max_gen=None, min_count=None):
    """Interpolate a list of (gens, vals) onto a common grid; return grid, mean, std.

    Runs are only averaged over the range they actually cover (beyond a run's
    last generation it contributes NaN, not a flat extrapolation). Grid points
    supported by fewer than ``min_count`` runs are dropped so the mean/band do
    not become dominated by one or two long outlier seeds.
    """
    if not series_list:
        return np.array([]), np.array([]), np.array([])
    hi = max(s[0][-1] for s in series_list) if max_gen is None else max_gen
    grid = np.arange(1, hi + 1)
    rows = []
    for g, v in series_list:
        row = np.interp(grid, g, v)
        row[grid > g[-1]] = np.nan  # do not extrapolate past a run's horizon
        rows.append(row)
    stacked = np.vstack(rows)
    counts = np.sum(~np.isnan(stacked), axis=0)
    if min_count is None:
        min_count = max(1, len(series_list) // 2)
    keep = counts >= min_count
    grid = grid[keep]
    stacked = stacked[:, keep]
    return grid, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def _plateau_stats(series_list, label):
    """Quantify plateau: fraction of final best reached by gen 750, and mean
    improvement per 100 gen before vs after PLATEAU_START."""
    frac_at_750 = []
    before, after = [], []
    for gens, vals in series_list:
        final = vals[-1]
        if final <= 0:
            continue
        v750 = np.interp(BUDGET_GENERATION, gens, vals)
        frac_at_750.append(v750 / final)
        # improvement per 100 gen before PLATEAU_START (from gen 100 to 700)
        v100 = np.interp(100, gens, vals)
        v700 = np.interp(PLATEAU_START, gens, vals)
        before.append((v700 - v100) / ((PLATEAU_START - 100) / 100))
        # after gen 700 up to end of run
        end_gen = gens[-1]
        if end_gen > PLATEAU_START:
            v_end = vals[-1]
            after.append((v_end - v700) / ((end_gen - PLATEAU_START) / 100))
    return {
        "label": label,
        "n": len(series_list),
        "frac_at_750_mean": float(np.mean(frac_at_750)) if frac_at_750 else float("nan"),
        "frac_at_750_min": float(np.min(frac_at_750)) if frac_at_750 else float("nan"),
        "impr_per100_before700_mean": float(np.mean(before)) if before else float("nan"),
        "impr_per100_after700_mean": float(np.mean(after)) if after else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.normpath(os.path.join(script_dir, "../../mlflow.db"))
    fig_dir = os.path.normpath(os.path.join(script_dir, "../../thesis/Figures"))
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"ERROR: mlflow.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- Long calibration runs, grouped by algorithm ---------------------
    long_neat, long_haneat = [], []
    long_run_report = defaultdict(lambda: defaultdict(int))
    for exp in LONG_EXPERIMENTS:
        runs = _load_best_so_far(cur, exp)
        for name, series in runs.items():
            cond = _condition(name)
            gens = series[0]
            long_run_report[exp][cond] += 1
            long_run_report[exp]["_maxgen"] = max(long_run_report[exp]["_maxgen"], int(gens[-1]))
            if cond == "neat":
                long_neat.append(series)
            else:  # ha_neat (no ablation in long runs)
                long_haneat.append(series)

    # --- Main-study pop200 runs, per condition ---------------------------
    main_runs = _load_best_so_far(cur, MAIN_EXPERIMENT)
    main_by_cond = defaultdict(list)
    for name, series in main_runs.items():
        main_by_cond[_condition(name)].append(series)

    conn.close()

    # ------------------------------------------------------------------
    # Plateau quantification
    # ------------------------------------------------------------------
    stats = []
    stats.append(_plateau_stats(long_neat, "Long runs NEAT (pop4096)"))
    stats.append(_plateau_stats(long_haneat, "Long runs HA-NEAT (pop4096)"))
    for cond in ("neat", "ha_neat", "ablation"):
        if main_by_cond[cond]:
            stats.append(_plateau_stats(main_by_cond[cond], f"Main study pop200 {cond}"))

    print("\n=== Runs used ===")
    for exp in LONG_EXPERIMENTS:
        r = long_run_report[exp]
        print(f"  {exp}: NEAT={r['neat']} HA-NEAT={r['ha_neat']} maxgen={r['_maxgen']}")
    print(f"  {MAIN_EXPERIMENT}: " + " ".join(
        f"{c}={len(main_by_cond[c])}" for c in ('neat', 'ha_neat', 'ablation')))

    print("\n=== Plateau quantification (best-so-far aggregated fitness) ===")
    hdr = f"{'group':32s} {'n':>3s} {'frac@750(mean)':>15s} {'frac@750(min)':>14s} " \
          f"{'impr/100 <700':>14s} {'impr/100 >700':>14s}"
    print(hdr)
    for s in stats:
        print(f"{s['label']:32s} {s['n']:3d} {s['frac_at_750_mean']:15.4f} "
              f"{s['frac_at_750_min']:14.4f} {s['impr_per100_before700_mean']:14.5f} "
              f"{s['impr_per100_after700_mean']:14.5f}")

    # ------------------------------------------------------------------
    # Figure: two panels
    #   (a) long calibration runs, full horizon, per-seed + mean band
    #   (b) main-study pop200 mean curves, truncated at 750
    # ------------------------------------------------------------------
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Panel (a): long runs ---
    for series in long_neat:
        ax1.plot(series[0], series[1], color=NEAT_COLOR, alpha=0.18, linewidth=0.7)
    for series in long_haneat:
        ax1.plot(series[0], series[1], color=HANEAT_COLOR, alpha=0.18, linewidth=0.7)

    g, m, sd = _mean_band(long_neat)
    if len(g):
        ax1.plot(g, m, color=NEAT_COLOR, linewidth=2.2, label=f"NEAT (n={len(long_neat)})")
        ax1.fill_between(g, m - sd, m + sd, color=NEAT_COLOR, alpha=0.15)
    g, m, sd = _mean_band(long_haneat)
    if len(g):
        ax1.plot(g, m, color=HANEAT_COLOR, linewidth=2.2, label=f"HA-NEAT (n={len(long_haneat)})")
        ax1.fill_between(g, m - sd, m + sd, color=HANEAT_COLOR, alpha=0.15)

    ax1.axvline(BUDGET_GENERATION, color="#333333", linestyle="--", linewidth=1.2)
    ymax1 = ax1.get_ylim()[1]
    ax1.text(BUDGET_GENERATION + 25, ax1.get_ylim()[0] + 0.02,
             "budget = 750 gen", rotation=90, va="bottom", ha="left",
             fontsize=8, color="#333333")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best-so-far aggregated fitness")
    ax1.set_title("(a) Calibration runs (population 4096)")
    ax1.legend(framealpha=0.9, edgecolor="none", loc="lower right")

    # --- Panel (b): main-study pop200 mean curves, truncated at budget ---
    cond_style = {
        "neat": (NEAT_COLOR, "NEAT"),
        "ha_neat": (HANEAT_COLOR, "HA-NEAT"),
        "ablation": (POP200_COLOR, "HA-NEAT (tanh only)"),
    }
    for cond in ("neat", "ha_neat", "ablation"):
        series_list = main_by_cond[cond]
        if not series_list:
            continue
        color, disp = cond_style[cond]
        g, m, sd = _mean_band(series_list, max_gen=BUDGET_GENERATION)
        ax2.plot(g, m, color=color, linewidth=2.2, label=f"{disp} (n={len(series_list)})")
        ax2.fill_between(g, m - sd, m + sd, color=color, alpha=0.15)

    ax2.axvline(BUDGET_GENERATION, color="#333333", linestyle="--", linewidth=1.2)
    ax2.set_xlim(0, BUDGET_GENERATION)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best-so-far aggregated fitness")
    ax2.set_title("(b) Main study (population 200)")
    ax2.legend(framealpha=0.9, edgecolor="none", loc="lower right")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(fig_dir, "budget_calibration.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
