"""Network complexity analysis of the pop200 final genomes (3 conditions, n=30 each).

For each saved champion genome (.npz) extract structural complexity:
  - n_nodes:        valid (non-NaN) rows in the nodes array. Includes the 17
                    input and 6 output nodes, which are always present.
  - n_hidden_nodes: valid nodes minus the 23 fixed input+output nodes.
  - n_connections:  valid (non-NaN) rows in the conns array. TensorNEAT deletes
                    disabled/removed connections outright (no `enabled` flag in
                    DefaultConn/OriginConn), so all valid connections are active.
  - HA-NEAT only:   activation-function distribution of hidden nodes
                    (node column 3 indexes [tanh, sigmoid, relu, sin, identity];
                    the sentinel -1 also means identity).

Array layout (verified on raw genomes):
  nodes (50, 4):  [index, bias, aggregation, activation]; NaN-padded rows.
  conns NEAT (200, 3):     [in, out, weight]                (DefaultConn)
  conns HA-NEAT (200, 4):  [in, out, historical_marker, weight] (OriginConn)

Statistics: Kruskal-Wallis omnibus across the 3 conditions, pairwise
Mann-Whitney U with Holm-Bonferroni correction, rank-biserial correlation and
Cliff's delta as effect sizes. Spearman correlation of complexity vs fitness
(normalized_min from analysis/outputs/pop200/combined.json), per condition
and pooled.

Reads:  results/multi_task_neat_vs_haneat_pop200/*.npz
        analysis/outputs/pop200/combined.json
Writes: analysis/outputs/pop200/network_complexity.md
        analysis/outputs/pop200/network_complexity.json
        analysis/outputs/pop200/complexity_hidden_nodes.png
        analysis/outputs/pop200/complexity_connections.png
        analysis/outputs/pop200/complexity_fitness_vs_connections.png
        analysis/outputs/pop200/haneat_activation_distribution.png

Usage:
    uv run python -m analysis.network_complexity_pop200
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.merge import collect_genomes  # noqa: E402

EXPERIMENT = "multi_task_neat_vs_haneat_pop200"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT)
COMBINED_JSON = os.path.join(PROJECT_ROOT, "analysis", "outputs", "pop200", "combined.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "outputs", "pop200")

N_INPUTS = 17
N_OUTPUTS = 6
N_FIXED = N_INPUTS + N_OUTPUTS  # node indices 0..22 are reserved

CONDITIONS = ["neat", "ha_neat", "ha_neat_ablation"]
COND_LABEL = {
    "neat": "NEAT",
    "ha_neat": "HA-NEAT",
    "ha_neat_ablation": "Ablation",
}
COND_COLOR = {
    "neat": "#4878CF",
    "ha_neat": "#D65F5F",
    "ha_neat_ablation": "#6ACC65",
}

# HA-NEAT activation options, in the order passed to BiasNode(activation_options=...)
ACT_NAMES = ["tanh", "sigmoid", "relu", "sin", "identity"]


# ---------------------------------------------------------------------------
# Genome extraction
# ---------------------------------------------------------------------------

def extract_complexity(path: str, algorithm: str) -> dict:
    data = np.load(path)
    nodes = np.asarray(data["nodes"])
    conns = np.asarray(data["conns"])

    node_valid = ~np.isnan(nodes[:, 0])
    conn_valid = ~np.isnan(conns[:, 0])

    n_nodes = int(node_valid.sum())
    n_conns = int(conn_valid.sum())

    idx = nodes[node_valid, 0].astype(int)
    hidden_mask = idx >= N_FIXED
    n_hidden = int(hidden_mask.sum())

    out = {
        "n_nodes": n_nodes,
        "n_hidden_nodes": n_hidden,
        "n_connections": n_conns,
    }

    if algorithm == "ha_neat":
        # activation index of hidden nodes; -1 sentinel means identity
        act = nodes[node_valid, 3][hidden_mask].astype(int)
        act = np.where(act == -1, ACT_NAMES.index("identity"), act)
        counts = {name: int((act == i).sum()) for i, name in enumerate(ACT_NAMES)}
        out["hidden_activation_counts"] = counts
    return out


def load_fitness_map() -> dict[str, float]:
    """file name -> normalized_min fitness (from the 20-episode re-evaluation)."""
    with open(COMBINED_JSON) as f:
        combined = json.load(f)
    fmap = {}
    for r in combined["runs"]:
        if "error" in r:
            continue
        n = r.get("normalized", {})
        if "hopper" in n and "walker2d" in n:
            fmap[r["file"]] = min(n["hopper"], n["walker2d"])
    return fmap


def collect_records() -> list[dict]:
    entries = collect_genomes(os.path.join(PROJECT_ROOT, "results"), [EXPERIMENT])
    fmap = load_fitness_map()
    records = []
    for e in entries:
        rec = {"file": e.fname, "condition": e.algorithm, "seed": e.seed}
        rec.update(extract_complexity(e.path, e.algorithm))
        rec["fitness_normalized_min"] = fmap.get(e.fname)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a) * len(b)))


def delta_magnitude(d: float) -> str:
    ad = abs(d)
    return ("negligible" if ad < 0.147 else "small" if ad < 0.33
            else "medium" if ad < 0.474 else "large")


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)),
        "median": float(np.median(x)), "q1": float(q1), "q3": float(q3),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def analyse_metric(groups: dict[str, np.ndarray]) -> dict:
    desc = {c: describe(groups[c]) for c in CONDITIONS}
    arrs = [groups[c] for c in CONDITIONS]
    H, p_kw = scipy_stats.kruskal(*arrs)
    N, k = sum(len(a) for a in arrs), len(arrs)
    eps2 = float((H - k + 1) / (N - k))

    pairs = [("neat", "ha_neat"), ("neat", "ha_neat_ablation"), ("ha_neat", "ha_neat_ablation")]
    pairwise, p_raw = {}, []
    for a, b in pairs:
        ga, gb = groups[a], groups[b]
        U, p = scipy_stats.mannwhitneyu(ga, gb, alternative="two-sided")
        rbc = float(1.0 - 2.0 * U / (len(ga) * len(gb)))  # rank-biserial (a<b positive)
        d = cliffs_delta(ga, gb)
        pairwise[f"{a}_vs_{b}"] = {
            "a": a, "b": b, "U": float(U), "p_mwu": float(p),
            "rank_biserial": rbc, "cliffs_delta": d, "magnitude": delta_magnitude(d),
        }
        p_raw.append(float(p))
    for (a, b), adj in zip(pairs, holm_bonferroni(p_raw)):
        pairwise[f"{a}_vs_{b}"]["p_holm"] = adj

    return {
        "descriptives": desc,
        "omnibus": {"kruskal_h": float(H), "p_kruskal": float(p_kw), "epsilon_sq": eps2},
        "pairwise": pairwise,
    }


def spearman_correlations(records: list[dict], metric: str) -> dict:
    out = {}
    for c in CONDITIONS + ["pooled"]:
        recs = [r for r in records
                if (c == "pooled" or r["condition"] == c)
                and r["fitness_normalized_min"] is not None]
        x = np.array([r[metric] for r in recs], float)
        y = np.array([r["fitness_normalized_min"] for r in recs], float)
        rho, p = scipy_stats.spearmanr(x, y)
        out[c] = {"n": len(x), "rho": float(rho), "p": float(p)}
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _box_strip(records, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    data = []
    for i, c in enumerate(CONDITIONS):
        vals = np.array([r[metric] for r in records if r["condition"] == c], float)
        data.append(vals)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=22, alpha=0.55,
                   color=COND_COLOR[c], edgecolors="none", zorder=3)
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, zorder=2)
    for patch, c in zip(bp["boxes"], CONDITIONS):
        patch.set(facecolor="white", edgecolor=COND_COLOR[c], linewidth=1.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.5)
    for elem in ("whiskers", "caps"):
        for line, c in zip(bp[elem], [x for c in CONDITIONS for x in (c, c)]):
            line.set(color=COND_COLOR[c])
    ax.set_xticks(range(1, len(CONDITIONS) + 1))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_fitness_scatter(records, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in CONDITIONS:
        recs = [r for r in records if r["condition"] == c and r["fitness_normalized_min"] is not None]
        x = [r["n_connections"] for r in recs]
        y = [r["fitness_normalized_min"] for r in recs]
        ax.scatter(x, y, s=30, alpha=0.7, color=COND_COLOR[c], label=COND_LABEL[c], edgecolors="none")
    ax.set_xlabel("Number of connections")
    ax.set_ylabel("Fitness (normalized_min, 20-episode re-eval)")
    ax.set_title("Fitness vs. network size (pop200 champions)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_activation_bars(records, fname):
    ha = sorted([r for r in records if r["condition"] == "ha_neat"], key=lambda r: r["seed"] or 0)
    act_colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66"]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(ha))
    bottom = np.zeros(len(ha))
    for i, name in enumerate(ACT_NAMES):
        vals = np.array([r["hidden_activation_counts"][name] for r in ha], float)
        ax.bar(xs, vals, bottom=bottom, color=act_colors[i], label=name, width=0.75)
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["seed"]) for r in ha], rotation=90, fontsize=7)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Hidden nodes")
    ax.set_title("HA-NEAT: activation functions of hidden nodes per seed (pop200)")
    ax.legend(frameon=False, ncol=5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def med_iqr(d):
    return f"{d['median']:.1f} [{d['q1']:.1f}, {d['q3']:.1f}]"


def metric_md(name, m):
    lines = [f"### {name}", "",
             "| Condition | n | Median [IQR] | Mean ± SD | Min–Max |",
             "|---|---|---|---|---|"]
    for c in CONDITIONS:
        d = m["descriptives"][c]
        lines.append(f"| {COND_LABEL[c]} | {d['n']} | {med_iqr(d)} | "
                     f"{d['mean']:.1f} ± {d['std']:.1f} | {d['min']:.0f}–{d['max']:.0f} |")
    o = m["omnibus"]
    lines += ["",
              f"**Kruskal-Wallis:** H = {o['kruskal_h']:.3f}, p = {o['p_kruskal']:.4f} "
              f"{_stars(o['p_kruskal'])}, ε² = {o['epsilon_sq']:.3f}", "",
              "| Contrast | U | p (MWU) | p (Holm) | Cliff's δ (mag) | Rank-biserial |",
              "|---|---|---|---|---|---|"]
    for key, e in m["pairwise"].items():
        lines.append(f"| {COND_LABEL[e['a']]} vs {COND_LABEL[e['b']]} | {e['U']:.1f} | "
                     f"{e['p_mwu']:.4f} {_stars(e['p_mwu'])} | {e['p_holm']:.4f} {_stars(e['p_holm'])} | "
                     f"{e['cliffs_delta']:.3f} ({e['magnitude']}) | {e['rank_biserial']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def corr_md(name, corr):
    lines = [f"### Spearman: fitness (normalized_min) vs {name}", "",
             "| Group | n | ρ | p |", "|---|---|---|---|"]
    for c in CONDITIONS + ["pooled"]:
        e = corr[c]
        label = COND_LABEL.get(c, "Pooled")
        lines.append(f"| {label} | {e['n']} | {e['rho']:.3f} | {e['p']:.4f} {_stars(e['p'])} |")
    lines.append("")
    return "\n".join(lines)


def build_report(records, stats, corrs, act_summary, plot_files):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    p = [
        "# pop200 — Network Complexity of Final Champion Genomes",
        "",
        f"_Generated {ts} by `analysis/network_complexity_pop200.py`._",
        "",
        "## Setup",
        "",
        f"- 90 champion genomes (`results/{EXPERIMENT}/*.npz`), 30 per condition.",
        "- **Counting rules (verified on raw arrays):** genome arrays are fixed-size "
        "(50 nodes × 4, 200 conns) with NaN-padded unused rows; only non-NaN rows are counted. "
        f"The node array *includes* the {N_INPUTS} input and {N_OUTPUTS} output nodes "
        f"(indices 0–{N_FIXED - 1}); `n_hidden_nodes` = valid nodes − {N_FIXED}. "
        "Connections have no `enabled` flag in TensorNEAT (disabled connections are deleted), "
        "so every valid connection is active.",
        "- Fitness = `normalized_min` from the 20-episode re-evaluation "
        "(`analysis/outputs/pop200/combined.json`).",
        "- Effect size: Cliff's δ (thresholds 0.147/0.33/0.474) and rank-biserial correlation.",
        "",
        "## Complexity by condition",
        "",
        metric_md("Hidden nodes", stats["n_hidden_nodes"]),
        metric_md("Connections", stats["n_connections"]),
        metric_md("Total nodes (incl. 17 inputs + 6 outputs)", stats["n_nodes"]),
        "## Complexity vs fitness",
        "",
        corr_md("n_connections", corrs["n_connections"]),
        corr_md("n_hidden_nodes", corrs["n_hidden_nodes"]),
        "## HA-NEAT hidden-node activation functions",
        "",
        act_summary,
        "",
        "## Figures",
        "",
        "\n".join(f"- `{os.path.basename(f)}`" for f in plot_files),
        "",
        interpretation(stats, corrs),
        "",
        "## Suggested further analyses",
        "",
        "- **Complexity over generations:** extract per-generation node/connection counts "
        "of the best genome from the MLflow metrics (or re-log them) to see whether HA-NEAT's "
        "structural growth *rate* differs even if the endpoints do not — bloat dynamics are "
        "invisible in final-champion snapshots.",
        "- **Species counts over time:** pull `species_count` per generation from `mlflow.db` "
        "to test whether activation diversity sustains more species (diversity maintenance) "
        "under the shared compat threshold of 0.3.",
        "- **Per-task fitness trade-off:** scatter norm_hopper vs norm_walker2d per champion and "
        "compare the trade-off frontier across conditions — conditions could reach equal "
        "normalized_min via different task balances.",
        "- **Activation-function enrichment vs fitness within HA-NEAT:** correlate the fraction "
        "of non-tanh hidden nodes (or activation entropy) with fitness across the 30 HA-NEAT "
        "seeds to test whether activation diversity is actually exploited by better solutions.",
        "",
    ]
    return "\n".join(p)


def interpretation(stats, corrs):
    hn = stats["n_hidden_nodes"]["omnibus"]
    cn = stats["n_connections"]["omnibus"]
    pooled_c = corrs["n_connections"]["pooled"]
    lines = ["## Interpretation", ""]

    def verdict(o):
        return "no significant difference" if o["p_kruskal"] >= 0.05 else "a significant difference"

    lines.append(
        f"- Hidden nodes: Kruskal-Wallis finds {verdict(hn)} across conditions "
        f"(H = {hn['kruskal_h']:.2f}, p = {hn['p_kruskal']:.3f}); connections likewise "
        f"{verdict(cn)} (H = {cn['kruskal_h']:.2f}, p = {cn['p_kruskal']:.3f})."
    )
    meds = {c: stats["n_hidden_nodes"]["descriptives"][c]["median"] for c in CONDITIONS}
    medc = {c: stats["n_connections"]["descriptives"][c]["median"] for c in CONDITIONS}
    lines.append(
        "- Median hidden nodes: " + ", ".join(f"{COND_LABEL[c]} = {meds[c]:.1f}" for c in CONDITIONS)
        + "; median connections: " + ", ".join(f"{COND_LABEL[c]} = {medc[c]:.1f}" for c in CONDITIONS) + "."
    )
    lines.append(
        f"- Pooled Spearman correlation between connections and fitness: "
        f"ρ = {pooled_c['rho']:.2f} (p = {pooled_c['p']:.3f})."
    )
    lines.append(
        "- Given that fitness itself showed no significant difference across conditions "
        "(Kruskal-Wallis p = 0.476, see `report.md`), the structural comparison addresses whether "
        "HA-NEAT reaches equivalent fitness with *different* complexity — e.g., fewer hidden "
        "nodes because activation diversity substitutes for structure. The tables above show "
        "whether that substitution effect is present at this population scale."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = collect_records()

    counts = {c: sum(1 for r in records if r["condition"] == c) for c in CONDITIONS}
    print(f"Collected {len(records)} genomes: {counts}")
    assert counts == {c: 30 for c in CONDITIONS}, f"Unexpected condition counts: {counts}"

    metrics = ["n_hidden_nodes", "n_connections", "n_nodes"]
    stats = {m: analyse_metric({c: np.array([r[m] for r in records if r["condition"] == c], float)
                                for c in CONDITIONS})
             for m in metrics}
    corrs = {m: spearman_correlations(records, m) for m in ("n_connections", "n_hidden_nodes")}

    # HA-NEAT activation summary
    ha = [r for r in records if r["condition"] == "ha_neat"]
    totals = {name: sum(r["hidden_activation_counts"][name] for r in ha) for name in ACT_NAMES}
    total_hidden = sum(totals.values())
    n_seeds_with_hidden = sum(1 for r in ha if r["n_hidden_nodes"] > 0)
    act_lines = [
        f"Across the 30 HA-NEAT seeds there are **{total_hidden} hidden nodes in total** "
        f"({n_seeds_with_hidden}/30 seeds evolved at least one hidden node).",
        "",
        "| Activation | Hidden nodes | Share |", "|---|---|---|",
    ]
    for name in ACT_NAMES:
        share = totals[name] / total_hidden if total_hidden else float("nan")
        act_lines.append(f"| {name} | {totals[name]} | {share:.1%} |")
    act_summary = "\n".join(act_lines)

    plot_files = [
        _box_strip(records, "n_hidden_nodes", "Hidden nodes",
                   "Hidden node count of final champions (pop200)", "complexity_hidden_nodes.png"),
        _box_strip(records, "n_connections", "Connections",
                   "Connection count of final champions (pop200)", "complexity_connections.png"),
        plot_fitness_scatter(records, "complexity_fitness_vs_connections.png"),
        plot_activation_bars(records, "haneat_activation_distribution.png"),
    ]

    report = build_report(records, stats, corrs, act_summary, plot_files)
    report_path = os.path.join(OUT_DIR, "network_complexity.md")
    with open(report_path, "w") as f:
        f.write(report)

    json_path = os.path.join(OUT_DIR, "network_complexity.json")
    with open(json_path, "w") as f:
        json.dump({"records": records, "stats": stats, "correlations": corrs,
                   "haneat_activation_totals": totals}, f, indent=2)

    # console summary
    for m in metrics:
        print(f"\n{m}:")
        for c in CONDITIONS:
            d = stats[m]["descriptives"][c]
            print(f"  {COND_LABEL[c]:10s} median={d['median']:.1f} IQR=[{d['q1']:.1f},{d['q3']:.1f}] "
                  f"mean={d['mean']:.1f}±{d['std']:.1f}")
        o = stats[m]["omnibus"]
        print(f"  KW: H={o['kruskal_h']:.3f} p={o['p_kruskal']:.4f} eps2={o['epsilon_sq']:.3f}")
        for k, e in stats[m]["pairwise"].items():
            print(f"    {k:34s} p={e['p_mwu']:.4f} holm={e['p_holm']:.4f} "
                  f"delta={e['cliffs_delta']:.3f} ({e['magnitude']})")
    for m, corr in corrs.items():
        print(f"\nSpearman fitness vs {m}:")
        for c, e in corr.items():
            print(f"  {c:18s} rho={e['rho']:.3f} p={e['p']:.4f} (n={e['n']})")
    print(f"\nHA-NEAT hidden activations: {totals} (total {total_hidden})")
    print(f"\nWrote:\n  {report_path}\n  {json_path}")
    for f_ in plot_files:
        print(f"  {f_}")


if __name__ == "__main__":
    main()
