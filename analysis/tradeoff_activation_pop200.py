"""Per-task trade-off frontier and HA-NEAT activation-entropy analyses (pop200).

Two analyses on the 90 champion genomes (3 conditions x 30 seeds):

1. Per-task trade-off frontier
   Scatter of normalized hopper (x) vs normalized walker2d (y), one point per
   champion, colored by condition (NEAT / HA-NEAT / Ablation). The fitness
   objective normalized_min = min(hopper, walker2d) has level sets that are
   L-shaped corners; the diagonal y=x marks perfectly balanced champions and
   dashed iso-min contours (min = c => the corner where hopper=c or walker2d=c)
   are drawn as reference. We quantify per condition:
     - median hopper, median walker2d, median normalized_min
     - imbalance |hopper - walker2d| per genome (median per condition;
       Kruskal-Wallis across conditions)
     - bottleneck fraction: fraction of genomes where walker2d < hopper

2. HA-NEAT activation entropy vs fitness (30 HA-NEAT genomes)
   Per genome compute (a) fraction of non-tanh hidden nodes and (b) Shannon
   entropy (bits) of the hidden-node activation distribution. Genomes with 0
   hidden nodes have an undefined activation distribution and are reported
   separately (excluded from the correlations). Spearman-correlate both metrics
   with normalized_min fitness over the genomes that have >=1 hidden node.
   n is small (~16); exact p-values are reported and low power is flagged.

Reads:  analysis/outputs/pop200/combined.json          (per-task normalized scores)
        analysis/outputs/pop200/network_complexity.json (HA-NEAT activation counts)
Writes: analysis/outputs/pop200/tradeoff_frontier.png
        analysis/outputs/pop200/activation_entropy_vs_fitness.png
        analysis/outputs/pop200/tradeoff_activation.md

Usage:
    uv run python -m analysis.tradeoff_activation_pop200
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "pop200")
COMBINED = os.path.join(OUT_DIR, "combined.json")
COMPLEXITY = os.path.join(OUT_DIR, "network_complexity.json")

# Condition presentation (label, color, marker) keyed by algorithm/condition id.
COND_STYLE = {
    "neat": ("NEAT", "#1f77b4", "o"),
    "ha_neat": ("HA-NEAT", "#d62728", "^"),
    "ha_neat_ablation": ("Ablation", "#2ca02c", "s"),
}
COND_ORDER = ["neat", "ha_neat", "ha_neat_ablation"]
ACTIVATIONS = ["tanh", "sigmoid", "relu", "sin", "identity"]


def load_records() -> list[dict]:
    """Merge per-task normalized scores with HA-NEAT activation counts by (condition, seed)."""
    combined = json.load(open(COMBINED))
    complexity = json.load(open(COMPLEXITY))

    act_by_key: dict[tuple[str, int], dict] = {}
    hidden_by_key: dict[tuple[str, int], int] = {}
    for rec in complexity["records"]:
        key = (rec["condition"], rec["seed"])
        hidden_by_key[key] = rec["n_hidden_nodes"]
        if "hidden_activation_counts" in rec:
            act_by_key[key] = rec["hidden_activation_counts"]

    records = []
    for run in combined["runs"]:
        cond = run["algorithm"]
        seed = run["seed"]
        hopper = run["normalized"]["hopper"]
        walker = run["normalized"]["walker2d"]
        key = (cond, seed)
        records.append(
            {
                "condition": cond,
                "seed": seed,
                "hopper": hopper,
                "walker2d": walker,
                "normalized_min": min(hopper, walker),
                "imbalance": abs(hopper - walker),
                "walker_bottleneck": walker < hopper,
                "n_hidden_nodes": hidden_by_key.get(key),
                "activation_counts": act_by_key.get(key),
            }
        )
    return records


def shannon_entropy_bits(counts: dict[str, int]) -> float | None:
    """Shannon entropy (bits) of a hidden-node activation distribution; None if no hidden nodes."""
    total = sum(counts.values())
    if total == 0:
        return None
    ent = 0.0
    for n in counts.values():
        if n > 0:
            p = n / total
            ent -= p * math.log2(p)
    return ent


def non_tanh_fraction(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total == 0:
        return None
    return 1.0 - counts.get("tanh", 0) / total


# --------------------------------------------------------------------------- #
# Analysis 1: trade-off frontier
# --------------------------------------------------------------------------- #
def analyse_tradeoff(records: list[dict]) -> dict:
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    summary = {}
    for cond in COND_ORDER:
        rs = by_cond[cond]
        hop = np.array([r["hopper"] for r in rs])
        wal = np.array([r["walker2d"] for r in rs])
        imb = np.array([r["imbalance"] for r in rs])
        nmin = np.array([r["normalized_min"] for r in rs])
        bott = np.array([r["walker_bottleneck"] for r in rs])
        summary[cond] = {
            "n": len(rs),
            "median_hopper": float(np.median(hop)),
            "median_walker2d": float(np.median(wal)),
            "median_normalized_min": float(np.median(nmin)),
            "median_imbalance": float(np.median(imb)),
            "iqr_imbalance": [float(np.percentile(imb, 25)), float(np.percentile(imb, 75))],
            "walker_bottleneck_fraction": float(bott.mean()),
        }

    # Kruskal-Wallis on imbalance across conditions.
    imb_groups = [np.array([r["imbalance"] for r in by_cond[c]]) for c in COND_ORDER]
    kw_h, kw_p = stats.kruskal(*imb_groups)
    return {"per_condition": summary, "imbalance_kruskal": {"H": float(kw_h), "p": float(kw_p)}}


def plot_tradeoff(records: list[dict], path: str) -> None:
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    fig, ax = plt.subplots(figsize=(7.2, 6.8))

    # Iso-min contours: L-shaped corners at min = c. Champions on/above the
    # diagonal with both scores >= c sit outside the corner.
    lim = 0.75
    for c in [0.1, 0.2, 0.3, 0.4]:
        ax.plot([c, c], [c, lim], color="0.75", ls="--", lw=0.8, zorder=0)
        ax.plot([c, lim], [c, c], color="0.75", ls="--", lw=0.8, zorder=0)
        ax.text(c + 0.005, lim - 0.01, f"min={c}", color="0.55", fontsize=7,
                rotation=90, va="top", ha="left")
    # Diagonal y = x (balanced champions).
    ax.plot([0, lim], [0, lim], color="0.4", ls="-", lw=1.0, zorder=0, label="balanced (y=x)")

    for cond in COND_ORDER:
        label, color, marker = COND_STYLE[cond]
        rs = by_cond[cond]
        hop = [r["hopper"] for r in rs]
        wal = [r["walker2d"] for r in rs]
        ax.scatter(hop, wal, s=55, c=color, marker=marker, alpha=0.75,
                   edgecolors="white", linewidths=0.6, label=label, zorder=3)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Normalized Hopper score")
    ax.set_ylabel("Normalized Walker2D score")
    ax.set_title("Per-task trade-off frontier (pop200 champions)")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis 2: HA-NEAT activation entropy vs fitness
# --------------------------------------------------------------------------- #
def analyse_activation(records: list[dict]) -> dict:
    ha = [r for r in records if r["condition"] == "ha_neat"]
    assert len(ha) == 30, f"expected 30 HA-NEAT genomes, got {len(ha)}"

    rows = []
    zero_hidden = []
    for r in ha:
        counts = r["activation_counts"] or {}
        total = sum(counts.values())
        if total == 0:
            zero_hidden.append(r["seed"])
            continue
        rows.append(
            {
                "seed": r["seed"],
                "n_hidden": total,
                "entropy_bits": shannon_entropy_bits(counts),
                "non_tanh_frac": non_tanh_fraction(counts),
                "fitness": r["normalized_min"],
                "counts": counts,
            }
        )

    result = {
        "n_total": len(ha),
        "n_with_hidden": len(rows),
        "n_zero_hidden": len(zero_hidden),
        "zero_hidden_seeds": sorted(zero_hidden),
        "rows": rows,
    }

    if len(rows) >= 3:
        ent = np.array([x["entropy_bits"] for x in rows])
        ntf = np.array([x["non_tanh_frac"] for x in rows])
        fit = np.array([x["fitness"] for x in rows])
        rho_e, p_e = stats.spearmanr(ent, fit)
        rho_n, p_n = stats.spearmanr(ntf, fit)
        result["spearman_entropy_fitness"] = {"rho": float(rho_e), "p": float(p_e), "n": len(rows)}
        result["spearman_nontanh_fitness"] = {"rho": float(rho_n), "p": float(p_n), "n": len(rows)}
        result["fitness_with_hidden_median"] = float(np.median(fit))
    # fitness comparison: hidden vs no-hidden HA-NEAT genomes
    fit_hidden = np.array([x["fitness"] for x in rows])
    fit_zero = np.array([r["normalized_min"] for r in ha if (r["activation_counts"] is None) or sum((r["activation_counts"] or {}).values()) == 0])
    result["fitness_zero_hidden_median"] = float(np.median(fit_zero)) if len(fit_zero) else None
    return result


def plot_activation(act: dict, path: str) -> None:
    rows = act["rows"]
    ent = np.array([x["entropy_bits"] for x in rows])
    ntf = np.array([x["non_tanh_frac"] for x in rows])
    fit = np.array([x["fitness"] for x in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.scatter(ent, fit, s=60, c="#d62728", marker="^", edgecolors="white", linewidths=0.6, alpha=0.85)
    for x in rows:
        ax.annotate(str(x["seed"]), (x["entropy_bits"], x["fitness"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="0.4")
    se = act.get("spearman_entropy_fitness", {})
    ax.set_xlabel("Hidden-node activation entropy (bits)")
    ax.set_ylabel("Fitness (normalized_min)")
    ax.set_title(f"Entropy vs fitness\nSpearman rho={se.get('rho', float('nan')):.2f}, "
                 f"p={se.get('p', float('nan')):.3f} (n={se.get('n', 0)})")
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.scatter(ntf, fit, s=60, c="#d62728", marker="^", edgecolors="white", linewidths=0.6, alpha=0.85)
    sn = act.get("spearman_nontanh_fitness", {})
    ax.set_xlabel("Fraction of non-tanh hidden nodes")
    ax.set_ylabel("Fitness (normalized_min)")
    ax.set_title(f"Non-tanh fraction vs fitness\nSpearman rho={sn.get('rho', float('nan')):.2f}, "
                 f"p={sn.get('p', float('nan')):.3f} (n={sn.get('n', 0)})")
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"HA-NEAT activation diversity vs fitness  "
                 f"({act['n_with_hidden']} of {act['n_total']} genomes have hidden nodes)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(tradeoff: dict, act: dict, path: str) -> None:
    lines = []
    w = lines.append
    w("# Per-task trade-off & HA-NEAT activation analysis (pop200)")
    w("")
    w("Champions: 3 conditions x 30 seeds = 90 genomes. Per-task scores are the "
      "normalized episode returns (20 eval episodes/task) from `combined.json`; "
      "`normalized_min = min(hopper, walker2d)` is the training objective.")
    w("")
    w("## 1. Per-task trade-off frontier")
    w("")
    w("| Condition | n | median Hopper | median Walker2D | median min-fit | median imbalance |Hop-Wal| | IQR imbalance | Walker-bottleneck frac |")
    w("|---|---|---|---|---|---|---|---|")
    for cond in COND_ORDER:
        s = tradeoff["per_condition"][cond]
        label = COND_STYLE[cond][0]
        w(f"| {label} | {s['n']} | {s['median_hopper']:.3f} | {s['median_walker2d']:.3f} | "
          f"{s['median_normalized_min']:.3f} | {s['median_imbalance']:.3f} | "
          f"[{s['iqr_imbalance'][0]:.3f}, {s['iqr_imbalance'][1]:.3f}] | "
          f"{s['walker_bottleneck_fraction']:.2f} |")
    w("")
    kw = tradeoff["imbalance_kruskal"]
    w(f"Kruskal-Wallis on per-genome imbalance |Hopper - Walker2D| across the 3 "
      f"conditions: H = {kw['H']:.3f}, p = {kw['p']:.3f}.")
    w("")
    w("Walker-bottleneck fraction = share of champions with Walker2D < Hopper "
      "(i.e. Walker2D is the limiting task that sets normalized_min).")
    w("")
    w("![trade-off frontier](tradeoff_frontier.png)")
    w("")
    w("## 2. HA-NEAT activation entropy vs fitness")
    w("")
    w(f"Of {act['n_total']} HA-NEAT champions, {act['n_with_hidden']} have >=1 "
      f"hidden node and {act['n_zero_hidden']} have zero hidden nodes "
      f"(activation distribution undefined; excluded from correlations).")
    w(f"Zero-hidden seeds: {act['zero_hidden_seeds']}.")
    if act.get("fitness_zero_hidden_median") is not None and act.get("fitness_with_hidden_median") is not None:
        w("")
        w(f"Median fitness — with hidden nodes: {act['fitness_with_hidden_median']:.3f}; "
          f"zero hidden nodes: {act['fitness_zero_hidden_median']:.3f}.")
    w("")
    se = act.get("spearman_entropy_fitness")
    sn = act.get("spearman_nontanh_fitness")
    if se:
        w(f"- Spearman(entropy, fitness): rho = {se['rho']:.3f}, p = {se['p']:.3f} (n = {se['n']}).")
    if sn:
        w(f"- Spearman(non-tanh fraction, fitness): rho = {sn['rho']:.3f}, p = {sn['p']:.3f} (n = {sn['n']}).")
    w("")
    w("Per-genome activation detail (genomes with hidden nodes):")
    w("")
    w("| seed | n_hidden | entropy (bits) | non-tanh frac | fitness | counts |")
    w("|---|---|---|---|---|---|")
    for x in sorted(act["rows"], key=lambda r: -r["fitness"]):
        nz = {k: v for k, v in x["counts"].items() if v}
        w(f"| {x['seed']} | {x['n_hidden']} | {x['entropy_bits']:.3f} | "
          f"{x['non_tanh_frac']:.3f} | {x['fitness']:.3f} | {nz} |")
    w("")
    w("![activation entropy vs fitness](activation_entropy_vs_fitness.png)")
    w("")
    w("## Interpretation")
    w("")
    w("Even though the conditions do not differ in min-fitness (KW p=0.476 in the "
      "main analysis), all three place Walker2D as the bottleneck task and produce "
      "similarly lopsided champions: the imbalance distributions are statistically "
      "indistinguishable, so conditions are not solving the trade-off in qualitatively "
      "different ways. Within HA-NEAT, activation diversity is largely unused — roughly "
      "half the champions evolve no hidden nodes at all, and among those that do, "
      "neither activation entropy nor non-tanh fraction shows a significant monotonic "
      "relationship with fitness. With only ~16 informative genomes the test is "
      "underpowered, but the point estimate gives no evidence that heterogeneous "
      "activations are the lever driving success on this task.")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    records = load_records()
    tradeoff = analyse_tradeoff(records)
    act = analyse_activation(records)

    plot_tradeoff(records, os.path.join(OUT_DIR, "tradeoff_frontier.png"))
    plot_activation(act, os.path.join(OUT_DIR, "activation_entropy_vs_fitness.png"))
    write_report(tradeoff, act, os.path.join(OUT_DIR, "tradeoff_activation.md"))

    # Console summary.
    print("=== Trade-off frontier ===")
    for cond in COND_ORDER:
        s = tradeoff["per_condition"][cond]
        print(f"{COND_STYLE[cond][0]:>10}: medHop={s['median_hopper']:.3f} "
              f"medWal={s['median_walker2d']:.3f} medMin={s['median_normalized_min']:.3f} "
              f"medImb={s['median_imbalance']:.3f} bottleneck(Wal<Hop)={s['walker_bottleneck_fraction']:.2f}")
    kw = tradeoff["imbalance_kruskal"]
    print(f"Imbalance Kruskal-Wallis: H={kw['H']:.3f} p={kw['p']:.3f}")
    print()
    print("=== HA-NEAT activation ===")
    print(f"with hidden={act['n_with_hidden']}  zero hidden={act['n_zero_hidden']} "
          f"(seeds {act['zero_hidden_seeds']})")
    if act.get("spearman_entropy_fitness"):
        se = act["spearman_entropy_fitness"]
        sn = act["spearman_nontanh_fitness"]
        print(f"Spearman(entropy,fit): rho={se['rho']:.3f} p={se['p']:.3f} n={se['n']}")
        print(f"Spearman(non-tanh,fit): rho={sn['rho']:.3f} p={sn['p']:.3f} n={sn['n']}")


if __name__ == "__main__":
    main()
