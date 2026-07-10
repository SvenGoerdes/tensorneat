#!/usr/bin/env python3
"""Aggregation-function calibration figure.

Justifies the choice of the normalized-MINIMUM fitness aggregation over
normalized SUM and normalized PRODUCT for the two-task study (Hopper +
Walker2D, single shared network).

The argument: under a SUM or PRODUCT aggregate, a genome can post a high
combined score while specializing on one task and collapsing on the other,
because a strong task can compensate for a weak one. Under a MINIMUM aggregate
the score equals the WEAKER task, so no such compensation is possible and
evolution is forced to lift both tasks together.

DATA SOURCE
-----------
The per-task scores are taken from the *re-evaluation* JSON
    eval/outputs/multi_task_neat_vs_haneat_mjx_backend_*.json
NOT from the MLflow per-generation metrics. Reason: the MLflow keys
`fitness/hopper` and `fitness/walker2d` are population-level statistics logged
per generation and do NOT reconstruct `fitness/max` for any aggregation mode
(verified: for a MIN run `fitness/max` exceeds min(normH, normW)); they are not
the per-task breakdown of the single best genome. The re-evaluation JSON
instead re-runs each saved best genome five times per task and records its own
per-task reward, so its `normalized.hopper` / `normalized.walker2d` fields ARE
the genuine per-task profile of the reported best genome. Normalization matches
the study: hopper / 3000, walker2d / 5000 (BRAX_REFERENCE_REWARDS).

RUNS INCLUDED
-------------
The calibration sweep (experiment `multi_task_neat_vs_haneat_mjx_backend`,
never named by codename in thesis text) is a single-seed (123) grid:
    3 aggregations {sum, min, product}
  x 2 algorithms   {NEAT, HA-NEAT}
  x 3 populations  {512, 1024, 2048}
  x 2 budgets      {100, 200 gen}
  = 36 best genomes, 12 per aggregation.
Both algorithms and all pop/budget cells are pooled per aggregation: the point
being demonstrated is a property of the *aggregation function*, not of the
algorithm.

HONESTY NOTE
------------
This is a single-seed calibration grid, so per-cell noise is large and pop/gen
are confounded with the aggregation axis. The effect is therefore a tendency,
not a clean separation. The figure reports it as such: the balance gap
|normH - normW| is smallest under MIN and the weaker-task score is highest
under MIN, and SUM/PRODUCT each contain blatant single-task-collapse runs
(the worst being a SUM run whose Hopper score exceeds 1.0 while Walker2D
sits near 0.17) that MIN structurally cannot reward.

Usage:
    uv run python text/figures/plot_aggregation_calibration.py
"""

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Colours consistent with the other thesis figures.
AGG_COLORS = {
    "sum": "#2176AE",       # blue
    "min": "#3F8F4F",       # green (the chosen aggregation)
    "product": "#D64933",   # red
}
AGG_LABELS = {
    "sum": "sum",
    "min": "minimum",
    "product": "product",
}
AGG_ORDER = ["sum", "min", "product"]

HOPPER_REF = 3000.0
WALKER_REF = 5000.0


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


def load_runs(json_path):
    """Return list of dicts: {agg, alg, pop, gen, h, w} from re-eval JSON."""
    with open(json_path) as fh:
        data = json.load(fh)
    rows = []
    for r in data["runs"]:
        f = r["file"]
        if "normalized_sum" in f:
            agg = "sum"
        elif "normalized_min" in f:
            agg = "min"
        elif "normalized_product" in f:
            agg = "product"
        else:
            continue
        m = re.search(r"pop(\d+)_gen(\d+)", f)
        pop, gen = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        rows.append({
            "agg": agg,
            "alg": r["algorithm"],
            "pop": pop,
            "gen": gen,
            "h": float(r["normalized"]["hopper"]),
            "w": float(r["normalized"]["walker2d"]),
        })
    return rows


def report(rows):
    print("=== per-run normalized per-task scores (re-evaluation) ===")
    hdr = f"{'agg':8s}{'alg':9s}{'pop':6s}{'gen':5s}{'normH':>8s}{'normW':>8s}{'weaker':>8s}{'gap':>8s}"
    print(hdr)
    for r in sorted(rows, key=lambda x: (AGG_ORDER.index(x["agg"]), x["alg"], x["pop"], x["gen"])):
        weaker = min(r["h"], r["w"])
        gap = abs(r["h"] - r["w"])
        print(f"{r['agg']:8s}{r['alg']:9s}{r['pop']:<6d}{r['gen']:<5d}"
              f"{r['h']:8.3f}{r['w']:8.3f}{weaker:8.3f}{gap:8.3f}")

    print("\n=== summary per aggregation (n=12 each) ===")
    print(f"{'agg':10s}{'mean gap':>10s}{'mean weaker':>13s}{'min weaker':>12s}{'max gap':>10s}")
    for agg in AGG_ORDER:
        sel = [r for r in rows if r["agg"] == agg]
        gaps = [abs(r["h"] - r["w"]) for r in sel]
        weak = [min(r["h"], r["w"]) for r in sel]
        print(f"{AGG_LABELS[agg]:10s}{np.mean(gaps):10.3f}{np.mean(weak):13.3f}"
              f"{np.min(weak):12.3f}{np.max(gaps):10.3f}")


def make_figure(rows, out_path):
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Panel (a): grouped bars, weaker vs stronger task per aggregation ----
    # For each aggregation, average the STRONGER and WEAKER task score across the
    # 12 runs. A large stronger/weaker split = the aggregate is being propped up
    # by one task (specialization). MIN should show the smallest split.
    x = np.arange(len(AGG_ORDER))
    width = 0.36
    stronger_mean, stronger_sd = [], []
    weaker_mean, weaker_sd = [], []
    for agg in AGG_ORDER:
        sel = [r for r in rows if r["agg"] == agg]
        strong = [max(r["h"], r["w"]) for r in sel]
        weak = [min(r["h"], r["w"]) for r in sel]
        stronger_mean.append(np.mean(strong)); stronger_sd.append(np.std(strong))
        weaker_mean.append(np.mean(weak)); weaker_sd.append(np.std(weak))

    ax1.bar(x - width / 2, stronger_mean, width, yerr=stronger_sd, capsize=3,
            color="#999999", label="stronger task", zorder=3,
            error_kw={"elinewidth": 1, "alpha": 0.7})
    ax1.bar(x + width / 2, weaker_mean, width, yerr=weaker_sd, capsize=3,
            color=[AGG_COLORS[a] for a in AGG_ORDER], label="weaker task", zorder=3,
            error_kw={"elinewidth": 1, "alpha": 0.7})
    ax1.set_xticks(x)
    ax1.set_xticklabels([AGG_LABELS[a] for a in AGG_ORDER])
    ax1.set_xlabel("Aggregation function")
    ax1.set_ylabel("Normalized per-task score")
    ax1.set_title("(a) Stronger vs. weaker task (mean $\\pm$ s.d., n=12)")
    ax1.set_ylim(0, None)
    ax1.legend(framealpha=0.9, edgecolor="none", loc="upper right")

    # ---- Panel (b): scatter normH vs normW, one point per run ----
    # Diagonal = balanced; near an axis = specialized. Makes the exploit visible.
    lim = 1.10
    ax2.plot([0, lim], [0, lim], color="#888888", linestyle="--",
             linewidth=1.0, zorder=1, label="balanced (H = W)")
    for agg in AGG_ORDER:
        sel = [r for r in rows if r["agg"] == agg]
        hs = [r["h"] for r in sel]
        ws = [r["w"] for r in sel]
        ax2.scatter(hs, ws, s=42, color=AGG_COLORS[agg], edgecolor="white",
                    linewidth=0.6, alpha=0.9, zorder=3, label=AGG_LABELS[agg])
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Normalized Hopper score")
    ax2.set_ylabel("Normalized Walker2D score")
    ax2.set_title("(b) Per-task profile of each best genome")
    ax2.legend(framealpha=0.9, edgecolor="none", loc="upper right", fontsize=8)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.normpath(os.path.join(script_dir, "../../eval/outputs"))
    fig_dir = os.path.normpath(os.path.join(script_dir, "../../thesis/Figures"))
    os.makedirs(fig_dir, exist_ok=True)

    # Pick the newest mjx_backend re-evaluation JSON.
    candidates = sorted(
        f for f in os.listdir(eval_dir)
        if f.startswith("multi_task_neat_vs_haneat_mjx_backend") and f.endswith(".json")
    )
    if not candidates:
        print("ERROR: no mjx_backend re-evaluation JSON in eval/outputs", file=sys.stderr)
        sys.exit(1)
    json_path = os.path.join(eval_dir, candidates[-1])
    print(f"Data source: {json_path}")

    rows = load_runs(json_path)
    if len(rows) != 36:
        print(f"WARNING: expected 36 runs, found {len(rows)}", file=sys.stderr)
    report(rows)

    out_path = os.path.join(fig_dir, "aggregation_calibration.png")
    make_figure(rows, out_path)


if __name__ == "__main__":
    main()
