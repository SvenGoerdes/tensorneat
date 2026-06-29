"""3-way statistical comparison + Markdown report for the pop200 experiment.

Conditions: neat, ha_neat, ha_neat_ablation (n=30 each).
Headline metric: normalized_min = min(norm_hopper, norm_walker2d) per seed.
Secondary metrics: hopper, walker2d (raw reward and normalized).

Statistics per metric:
  - Descriptives: mean, std, SEM, 95% CI (t-based), median, IQR, min, max
  - Omnibus: Kruskal-Wallis across all three conditions
  - Pairwise (3 contrasts): Mann-Whitney U, Monte-Carlo permutation test,
    Cohen's d (+ magnitude), with Holm-Bonferroni correction across the
    family of 3 pairwise p-values (per test type, per metric family).

Reads:  analysis/outputs/pop200/combined.json
Writes: analysis/outputs/pop200/report.md
        analysis/outputs/pop200/stats.json
        analysis/outputs/pop200/stats.txt

Usage:
    uv run python -m analysis.report_pop200
    uv run python -m analysis.report_pop200 --input <combined.json> --outdir <dir>
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "text", "figures"))

from statistical_test import permutation_test_mc, cohens_d  # noqa: E402

REFERENCE_REWARDS = {"hopper": 3000, "walker2d": 5000}
TASKS = ["hopper", "walker2d"]
CONDITIONS = ["neat", "ha_neat", "ha_neat_ablation"]
COND_LABEL = {
    "neat": "NEAT",
    "ha_neat": "HA-NEAT",
    "ha_neat_ablation": "HA-NEAT (ablation, tanh-only)",
}
# Short, unambiguous labels for contrast tables and inline prose (the full
# COND_LABEL collides on "HA-NEAT" once the parenthetical is stripped).
COND_SHORT = {
    "neat": "NEAT",
    "ha_neat": "HA-NEAT",
    "ha_neat_ablation": "Ablation",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_runs(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    runs_ok = [r for r in data["runs"] if "error" not in r]
    by_cond = {c: [r for r in runs_ok if r.get("algorithm") == c] for c in CONDITIONS}
    return {"meta": data.get("metadata", {}), "by_cond": by_cond,
            "n_errors": len(data["runs"]) - len(runs_ok)}


def seed_means(runs: list[dict], task: str) -> np.ndarray:
    return np.array([r["tasks"][task]["mean"] for r in runs if task in r.get("tasks", {})])


def normalized_min(runs: list[dict]) -> np.ndarray:
    out = []
    for r in runs:
        n = r.get("normalized", {})
        if "hopper" in n and "walker2d" in n:
            out.append(min(n["hopper"], n["walker2d"]))
    return np.array(out)


def training_fitness(runs: list[dict]) -> np.ndarray:
    return np.array([r["training_fitness"] for r in runs
                     if r.get("training_fitness") is not None])


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "sem", "ci_lo", "ci_hi", "median", "iqr", "min", "max")}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    sem = float(std / np.sqrt(n)) if n > 1 else float("nan")
    if n > 1:
        tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_lo, ci_hi = mean - tcrit * sem, mean + tcrit * sem
    else:
        ci_lo = ci_hi = float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "median": float(np.median(x)), "iqr": float(q3 - q1),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


# ---------------------------------------------------------------------------
# Inferential statistics
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def pairwise_tests(groups: dict[str, np.ndarray]) -> dict:
    """All 3 pairwise contrasts. Returns dict keyed 'a_vs_b' plus Holm-corrected p's."""
    pairs = [("neat", "ha_neat"), ("neat", "ha_neat_ablation"), ("ha_neat", "ha_neat_ablation")]
    out = {}
    p_mwu, p_perm = [], []
    for a, b in pairs:
        ga, gb = groups[a], groups[b]
        U, pmw = scipy_stats.mannwhitneyu(ga, gb, alternative="two-sided")
        diff, pp, n_perms = permutation_test_mc(ga, gb)
        d = cohens_d(ga, gb)
        mag = ("negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5
               else "medium" if abs(d) < 0.8 else "large")
        key = f"{a}_vs_{b}"
        out[key] = {
            "a": a, "b": b,
            "mannwhitney_u": float(U), "p_mannwhitney": float(pmw),
            "observed_diff": float(diff), "p_permutation": float(pp), "n_permutations": int(n_perms),
            "cohens_d": float(d), "magnitude": mag,
        }
        p_mwu.append(float(pmw))
        p_perm.append(float(pp))

    adj_mwu = holm_bonferroni(p_mwu)
    adj_perm = holm_bonferroni(p_perm)
    for i, (a, b) in enumerate(pairs):
        key = f"{a}_vs_{b}"
        out[key]["p_mannwhitney_holm"] = adj_mwu[i]
        out[key]["p_permutation_holm"] = adj_perm[i]
    return out


def analyse_metric(groups: dict[str, np.ndarray]) -> dict:
    """Descriptives + omnibus + pairwise for one metric (3 conditions)."""
    desc = {c: describe(groups[c]) for c in CONDITIONS}
    arrs = [groups[c] for c in CONDITIONS if len(groups[c]) > 0]
    if len(arrs) >= 2 and all(len(a) >= 1 for a in arrs):
        H, p_kw = scipy_stats.kruskal(*arrs)
        # epsilon-squared effect size for Kruskal-Wallis
        N = sum(len(a) for a in arrs)
        k = len(arrs)
        eps2 = float((H - k + 1) / (N - k)) if N > k else float("nan")
    else:
        H, p_kw, eps2 = float("nan"), float("nan"), float("nan")
    return {
        "descriptives": desc,
        "omnibus": {"kruskal_h": float(H), "p_kruskal": float(p_kw), "epsilon_sq": eps2},
        "pairwise": pairwise_tests(groups),
    }


def build_stats(data: dict) -> dict:
    by_cond = data["by_cond"]
    metrics = {}

    # Headline: normalized_min
    metrics["normalized_min"] = analyse_metric(
        {c: normalized_min(by_cond[c]) for c in CONDITIONS}
    )
    # Secondary: per-task raw reward
    for task in TASKS:
        metrics[task] = analyse_metric(
            {c: seed_means(by_cond[c], task) for c in CONDITIONS}
        )
    # Training fitness (sanity vs re-eval)
    metrics["training_fitness"] = analyse_metric(
        {c: training_fitness(by_cond[c]) for c in CONDITIONS}
    )
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f(x, p=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def _stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def descriptives_table(metric: dict, scale: float = 1.0, unit: str = "") -> str:
    rows = ["| Condition | n | Mean | SD | SEM | 95% CI | Median | IQR | Min | Max |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        d = metric["descriptives"][c]
        ci = f"[{_f(d['ci_lo']*scale,3)}, {_f(d['ci_hi']*scale,3)}]"
        rows.append(
            f"| {COND_LABEL[c]} | {d['n']} | {_f(d['mean']*scale,3)} | {_f(d['std']*scale,3)} | "
            f"{_f(d['sem']*scale,3)} | {ci} | {_f(d['median']*scale,3)} | {_f(d['iqr']*scale,3)} | "
            f"{_f(d['min']*scale,3)} | {_f(d['max']*scale,3)} |"
        )
    return "\n".join(rows)


def omnibus_line(metric: dict) -> str:
    o = metric["omnibus"]
    return (f"**Kruskal-Wallis omnibus:** H = {_f(o['kruskal_h'],3)}, "
            f"p = {_f(o['p_kruskal'])} {_stars(o['p_kruskal'])}, "
            f"ε² = {_f(o['epsilon_sq'],3)}")


def pairwise_table(metric: dict, scale: float = 1.0) -> str:
    rows = ["| Contrast | Mean diff | Cohen's d (mag) | MWU U | p (MWU) | p Holm | Perm p | Perm Holm |",
            "|---|---|---|---|---|---|---|---|"]
    pw = metric["pairwise"]
    for key in ("neat_vs_ha_neat", "neat_vs_ha_neat_ablation", "ha_neat_vs_ha_neat_ablation"):
        e = pw[key]
        label = f"{COND_SHORT[e['a']]} vs {COND_SHORT[e['b']]}"
        rows.append(
            f"| {label} | {_f(e['observed_diff']*scale,3)} | "
            f"{_f(e['cohens_d'],3)} ({e['magnitude']}) | {_f(e['mannwhitney_u'],1)} | "
            f"{_f(e['p_mannwhitney'])} {_stars(e['p_mannwhitney'])} | "
            f"{_f(e['p_mannwhitney_holm'])} {_stars(e['p_mannwhitney_holm'])} | "
            f"{_f(e['p_permutation'])} | {_f(e['p_permutation_holm'])} |"
        )
    return "\n".join(rows)


def metric_section(title: str, metric: dict, scale: float, note: str = "") -> str:
    parts = [f"### {title}", ""]
    if note:
        parts += [note, ""]
    parts += [descriptives_table(metric, scale=scale), "",
              omnibus_line(metric), "",
              "Pairwise contrasts (Holm-Bonferroni corrected across the 3 contrasts):", "",
              pairwise_table(metric, scale=scale), ""]
    return "\n".join(parts)


def interpret(stats: dict) -> str:
    nm = stats["normalized_min"]
    o = nm["omnibus"]
    desc = nm["descriptives"]
    best = max(CONDITIONS, key=lambda c: desc[c]["mean"])
    lines = ["## Interpretation", ""]

    if np.isnan(o["p_kruskal"]):
        lines.append("- Omnibus test could not be computed.")
    elif o["p_kruskal"] >= 0.05:
        lines.append(
            f"- On the headline metric (`normalized_min`), the Kruskal-Wallis omnibus test finds "
            f"**no significant difference** among the three conditions "
            f"(H = {o['kruskal_h']:.2f}, p = {o['p_kruskal']:.3f}). "
            f"The three algorithms perform equivalently on the multi-task objective they were trained on."
        )
    else:
        lines.append(
            f"- The omnibus test is **significant** (H = {o['kruskal_h']:.2f}, p = {o['p_kruskal']:.3f}); "
            f"see the Holm-corrected pairwise contrasts to localize the difference."
        )

    means = ", ".join(f"{COND_SHORT[c]} = {desc[c]['mean']:.3f}" for c in CONDITIONS)
    lines.append(f"- Mean `normalized_min` by condition: {means} (highest: {COND_SHORT[best]}).")

    # Pairwise summary on normalized_min
    any_sig = False
    for key, e in nm["pairwise"].items():
        if not np.isnan(e["p_mannwhitney_holm"]) and e["p_mannwhitney_holm"] < 0.05:
            any_sig = True
            lines.append(
                f"- {COND_SHORT[e['a']]} vs {COND_SHORT[e['b']]}: "
                f"Holm-corrected p = {e['p_mannwhitney_holm']:.3f}, Cohen's d = {e['cohens_d']:.2f} ({e['magnitude']})."
            )
    if not any_sig:
        lines.append(
            "- After Holm-Bonferroni correction, **none** of the three pairwise contrasts on "
            "`normalized_min` reach significance. Effect sizes are summarized in the table above."
        )

    # Ablation-specific takeaway
    abl = nm["pairwise"]["ha_neat_vs_ha_neat_ablation"]
    lines.append(
        f"- **Ablation read:** HA-NEAT vs HA-NEAT(tanh-only) differ by d = {abl['cohens_d']:.2f} "
        f"({abl['magnitude']}), Holm p = {_f(abl['p_mannwhitney_holm'],3)}. "
        f"This isolates whether activation-function *diversity* (not the HA-NEAT machinery itself) "
        f"is the active ingredient."
    )
    lines.append("")
    return "\n".join(lines)


def build_report(stats: dict, data: dict, input_path: str) -> str:
    meta = data["meta"]
    by_cond = data["by_cond"]
    n_eval = meta.get("n_eval", "?")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    inv = " / ".join(f"{COND_SHORT[c]} = {len(by_cond[c])}" for c in CONDITIONS)

    p = [
        "# pop200 — Three-Way Algorithm Comparison (NEAT vs HA-NEAT vs Ablation)",
        "",
        f"_Generated {ts}. Source: `{os.path.relpath(input_path, PROJECT_ROOT)}`._",
        "",
        "## Setup",
        "",
        f"- **Experiment:** `multi_task_neat_vs_haneat_pop200` (pop = 200, gen = 750, "
        f"species_size = 10, compat = 0.3, aggregation = `normalized_min`, MJX backend).",
        f"- **Conditions (seeds re-evaluated):** {inv}.",
        f"- **Re-evaluation:** {n_eval} independent episodes per genome per task on "
        f"hopper + walker2d (MJX), seeds distinct from training rollouts.",
        f"- **Reference rewards (normalization):** hopper = {REFERENCE_REWARDS['hopper']}, "
        f"walker2d = {REFERENCE_REWARDS['walker2d']}.",
        f"- **Headline metric:** `normalized_min` = min(reward_hopper / {REFERENCE_REWARDS['hopper']}, "
        f"reward_walker2d / {REFERENCE_REWARDS['walker2d']}) — the multi-task objective optimized during training.",
        "",
        "Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant.",
        "",
        "## Headline metric: normalized_min",
        "",
        metric_section("normalized_min (per-seed)", stats["normalized_min"], scale=1.0),
        "## Secondary: per-task performance",
        "",
        metric_section("Hopper (raw reward)", stats["hopper"], scale=1.0,
                       note=f"Normalized by {REFERENCE_REWARDS['hopper']}."),
        metric_section("Walker2D (raw reward)", stats["walker2d"], scale=1.0,
                       note=f"Normalized by {REFERENCE_REWARDS['walker2d']}."),
        "## Sanity check: training fitness vs re-evaluation",
        "",
        "Training fitness is the single `normalized_min` value stored in each `.npz` at the end of "
        "training (one rollout). The re-evaluated `normalized_min` above averages "
        f"{n_eval} fresh episodes. Close agreement indicates the saved champions generalize across seeds.",
        "",
        metric_section("Training fitness (stored normalized_min)", stats["training_fitness"], scale=1.0),
        interpret(stats),
    ]
    return "\n".join(p)


def build_txt(stats: dict) -> str:
    import io
    buf = io.StringIO()
    pr = lambda *a: print(*a, file=buf)
    pr("pop200 3-way statistical comparison")
    pr("=" * 70)
    for mname, metric in stats.items():
        pr(f"\n{'='*70}\nMETRIC: {mname}\n{'='*70}")
        for c in CONDITIONS:
            d = metric["descriptives"][c]
            pr(f"  {COND_LABEL[c]:35s} n={d['n']}  mean={_f(d['mean'])}  sd={_f(d['std'])}  "
               f"sem={_f(d['sem'])}  95%CI=[{_f(d['ci_lo'])},{_f(d['ci_hi'])}]")
        o = metric["omnibus"]
        pr(f"  Kruskal-Wallis: H={_f(o['kruskal_h'],3)} p={_f(o['p_kruskal'])} eps2={_f(o['epsilon_sq'],3)}")
        for key, e in metric["pairwise"].items():
            pr(f"  {key:38s} d={_f(e['cohens_d'],3)} ({e['magnitude']:10s}) "
               f"p_mwu={_f(e['p_mannwhitney'])} holm={_f(e['p_mannwhitney_holm'])} "
               f"p_perm={_f(e['p_permutation'])} holm={_f(e['p_permutation_holm'])}")
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJECT_ROOT, "analysis", "outputs", "pop200", "combined.json"))
    ap.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "analysis", "outputs", "pop200"))
    args = ap.parse_args()

    data = load_runs(args.input)
    stats = build_stats(data)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.outdir, "stats.txt"), "w") as f:
        f.write(build_txt(stats))
    report = build_report(stats, data, args.input)
    with open(os.path.join(args.outdir, "report.md"), "w") as f:
        f.write(report)

    print(build_txt(stats))
    print(f"\nWrote:\n  {args.outdir}/report.md\n  {args.outdir}/stats.json\n  {args.outdir}/stats.txt")


if __name__ == "__main__":
    main()
