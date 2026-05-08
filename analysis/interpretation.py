"""Generate a thesis-ready markdown interpretation from stats + eval results."""

import os
import math
from datetime import datetime


REFERENCE_REWARDS = {"hopper": 3000, "walker2d": 5000}
TASKS = ["hopper", "walker2d"]


def _sig(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return f"p = {p:.3f} **"
    if p < 0.05:
        return f"p = {p:.3f} *"
    return f"p = {p:.3f} (n.s.)"


def _direction(r: dict, label_a: str, label_b: str) -> str:
    if math.isnan(r["mean_a"]) or math.isnan(r["mean_b"]):
        return "inconclusive"
    diff = r["mean_a"] - r["mean_b"]
    winner = label_a if diff > 0 else label_b
    return f"{winner} higher by {abs(diff):.1f} reward units"


def generate_markdown(
    test_results: dict,
    combined_json: dict,
    label_a: str,
    label_b: str,
    n_eval: int,
    backend: str,
    provenance: list[dict],
) -> str:
    summary = combined_json.get("summary", {})
    n_a = test_results.get("hopper", {}).get("n_a", "?")
    n_b = test_results.get("hopper", {}).get("n_b", "?")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Final Experiment Analysis — {ts}",
        "",
        "## Setup",
        "",
        f"- **Algorithms:** {label_a.upper()} (n={n_a} seeds) vs {label_b.upper()} (n={n_b} seeds)",
        f"- **Environments:** Hopper (11 obs, 3 act) + Walker2D (17 obs, 6 act)",
        f"- **Episodes per genome per task:** {n_eval}",
        f"- **Simulation backend:** {backend}",
        f"- **Population:** 4096 | **Generations:** 2000 | **Fitness:** normalized_min",
        "",
        "## Per-Task Results",
        "",
        "| Task | NEAT mean ± SEM | HA-NEAT mean ± SEM | NEAT norm | HA-NEAT norm | Cohen's d | p (MW) | p (perm) |",
        "|------|-----------------|---------------------|-----------|--------------|-----------|--------|----------|",
    ]

    for task in TASKS:
        r = test_results.get(task, {})
        ref = REFERENCE_REWARDS.get(task, 1.0)
        row = (
            f"| {task.capitalize()} "
            f"| {r.get('mean_a', float('nan')):.1f} ± {r.get('sem_a', float('nan')):.1f} "
            f"| {r.get('mean_b', float('nan')):.1f} ± {r.get('sem_b', float('nan')):.1f} "
            f"| {r.get('mean_norm_a', float('nan')):.3f} "
            f"| {r.get('mean_norm_b', float('nan')):.3f} "
            f"| {r.get('cohens_d', float('nan')):.3f} ({r.get('magnitude', '?')}) "
            f"| {_sig(r.get('p_mannwhitney'))} "
            f"| {_sig(r.get('p_permutation'))} |"
        )
        lines.append(row)

    # Aggregate
    r_agg = test_results.get("normalized_min", {})
    lines.append(
        f"| **norm_min** "
        f"| {r_agg.get('mean_a', float('nan')):.3f} ± {r_agg.get('sem_a', float('nan')):.3f} "
        f"| {r_agg.get('mean_b', float('nan')):.3f} ± {r_agg.get('sem_b', float('nan')):.3f} "
        f"| — | — "
        f"| {r_agg.get('cohens_d', float('nan')):.3f} ({r_agg.get('magnitude', '?')}) "
        f"| {_sig(r_agg.get('p_mannwhitney'))} "
        f"| {_sig(r_agg.get('p_permutation'))} |"
    )

    lines += ["", "## Per-Task Interpretation", ""]

    for task in TASKS:
        r = test_results.get(task, {})
        ref = REFERENCE_REWARDS.get(task, 1.0)
        sig_mw = r.get("p_mannwhitney") or 1.0
        direction = _direction(r, label_a.upper(), label_b.upper())
        sig_word = "statistically significant" if sig_mw < 0.05 else "not statistically significant"
        d = r.get("cohens_d", 0.0)
        mag = r.get("magnitude", "negligible")

        lines += [
            f"### {task.capitalize()}",
            "",
            f"On the {task} task, {direction}. "
            f"The effect size is {mag} (Cohen's d = {d:.3f}). "
            f"Mann-Whitney U: {_sig(r.get('p_mannwhitney'))}; "
            f"exact permutation: {_sig(r.get('p_permutation'))}. "
            f"The difference is **{sig_word}** at α = 0.05. "
            f"This is {'consistent' if sig_mw >= 0.05 else 'inconsistent'} with the ablation experiment "
            f"(`multi_task_neat_vs_haneat_ablation`), which found no statistically significant difference "
            f"between NEAT and HA-NEAT (tanh only) under smaller scale (pop=1024, gen=500).",
            "",
        ]

    # Aggregate interpretation
    r_agg = test_results.get("normalized_min", {})
    sig_agg = r_agg.get("p_mannwhitney") or 1.0
    lines += [
        "### Aggregate (normalized_min)",
        "",
        f"The headline multi-task metric min(norm_hopper, norm_walker2d) shows: "
        f"{label_a.upper()} = {r_agg.get('mean_a', float('nan')):.3f} ± {r_agg.get('sem_a', float('nan')):.3f}, "
        f"{label_b.upper()} = {r_agg.get('mean_b', float('nan')):.3f} ± {r_agg.get('sem_b', float('nan')):.3f}. "
        f"Cohen's d = {r_agg.get('cohens_d', float('nan')):.3f} ({r_agg.get('magnitude', '?')}), "
        f"{_sig(r_agg.get('p_mannwhitney'))}.",
        "",
    ]

    lines += [
        "## Caveats",
        "",
        f"- Small sample sizes (NEAT n={n_a}, HA-NEAT n={n_b}) limit statistical power; "
        "interpret p-values with caution.",
        f"- Each genome evaluated on {n_eval} stochastic episodes per task; "
        "variance across episodes contributes to uncertainty.",
        "- max_step = 1000 per episode; longer horizons may favour different architectures.",
        "- Any genomes that raised exceptions during re-evaluation were excluded from statistics.",
        "- HA-NEAT genomes pooled across `multi_task_neat_vs_haneat_final` and `multi_task_haneat_finalv2`; "
        "duplicates removed by file size (larger = more training).",
        "",
        "## HA-NEAT Genome Provenance",
        "",
        "| File | Experiment | Seed | Compat |",
        "|------|-----------|------|--------|",
    ]

    for p in provenance:
        lines.append(f"| {p.get('file','?')} | {p.get('experiment','?')} | {p.get('seed','?')} | {p.get('compat','?')} |")

    lines += [
        "",
        "## Drop-in Paragraph for results_and_discussion.md",
        "",
        "> We evaluated the best genome from each of the 10 NEAT and "
        f"{n_b} HA-NEAT independent runs using {n_eval} stochastic episodes per task "
        f"on the MJX physics backend (max_step=1000). "
        f"On Hopper, {_direction(test_results.get('hopper', {}), 'NEAT', 'HA-NEAT')} "
        f"(Cohen's d = {test_results.get('hopper', {}).get('cohens_d', float('nan')):.3f}, "
        f"{_sig(test_results.get('hopper', {}).get('p_mannwhitney'))}). "
        f"On Walker2D, {_direction(test_results.get('walker2d', {}), 'NEAT', 'HA-NEAT')} "
        f"(Cohen's d = {test_results.get('walker2d', {}).get('cohens_d', float('nan')):.3f}, "
        f"{_sig(test_results.get('walker2d', {}).get('p_mannwhitney'))}). "
        "These results are consistent with the ablation experiment, which found no significant advantage "
        "from heterogeneous activation functions under controlled conditions.",
    ]

    return "\n".join(lines) + "\n"


def write_interpretation(md: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "interpretation.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"  Saved: {path}")
