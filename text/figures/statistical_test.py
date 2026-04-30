#!/usr/bin/env python3
"""
Statistical comparison of algorithm performance from re-evaluation JSON.

Runs:
  1. Mann-Whitney U on seed-level means (conservative, n=3 vs n=3)
  2. Permutation test on seed-level means (exact, handles small n)
  3. Welch's t-test on all episodes (liberal, assumes independence)
  4. Effect size (Cohen's d) on seed-level means

Usage:
    python statistical_test.py --input ../../eval/outputs/multi_task_neat_vs_haneat_ablation_*.json
"""

import argparse
import json

import numpy as np
from scipy import stats


REFERENCE_REWARDS = {"hopper": 3000, "walker2d": 5000, "ant": 6000, "halfcheetah": 8000, "humanoid": 6000, "reacher": 5}


def detect_algorithm(run: dict) -> str:
    fname = run.get("file", "")
    if fname.startswith("ha_neat_ablation"):
        return "ha_neat_ablation"
    elif fname.startswith("ha_neat"):
        return "ha_neat"
    return "neat"


def permutation_test_mc(group_a, group_b, n_perms: int = 10_000, seed: int = 42):
    """Monte Carlo permutation test for difference in means (two-sided).

    C(40,20) ≈ 137B makes a true exact test infeasible at n=20 per arm.
    10k resamples gives a standard error on p of ~0.002 at p=0.05, which is
    more than sufficient for reporting purposes.
    """
    rng = np.random.default_rng(seed)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(combined)
        diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
        if abs(diff) >= abs(observed_diff):
            count_extreme += 1

    return observed_diff, count_extreme / n_perms, n_perms


def cohens_d(group_a, group_b):
    """Cohen's d effect size."""
    na, nb = len(group_a), len(group_b)
    pooled_std = np.sqrt(((na - 1) * np.var(group_a, ddof=1) + (nb - 1) * np.var(group_b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of algorithms.")
    parser.add_argument("--input", required=True, help="Path to evaluation JSON")
    parser.add_argument("--tasks", default="hopper,walker2d")
    parser.add_argument("--algo_a", default="neat", help="First algorithm (reference)")
    parser.add_argument("--algo_b", default="ha_neat_ablation", help="Second algorithm")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    with open(args.input) as f:
        data = json.load(f)

    runs = [r for r in data["runs"] if "error" not in r]
    n_eval = data["metadata"]["n_eval"]

    # Group by algorithm
    algo_runs = {}
    for r in runs:
        algo = detect_algorithm(r)
        algo_runs.setdefault(algo, []).append(r)

    print(f"Algorithms: {args.algo_a} (n={len(algo_runs.get(args.algo_a, []))}) vs "
          f"{args.algo_b} (n={len(algo_runs.get(args.algo_b, []))})")
    print(f"Episodes per seed: {n_eval}")
    print()

    for task in tasks:
        ref = REFERENCE_REWARDS.get(task, 1.0)
        print(f"{'='*70}")
        print(f"  Task: {task.upper()} (reference reward: {ref})")
        print(f"{'='*70}")

        # Collect seed-level means and all episodes
        seed_means_a, seed_means_b = [], []
        all_eps_a, all_eps_b = [], []

        for r in algo_runs.get(args.algo_a, []):
            if task in r["tasks"]:
                eps = r["tasks"][task]["episodes"]
                seed_means_a.append(np.mean(eps))
                all_eps_a.extend(eps)

        for r in algo_runs.get(args.algo_b, []):
            if task in r["tasks"]:
                eps = r["tasks"][task]["episodes"]
                seed_means_b.append(np.mean(eps))
                all_eps_b.extend(eps)

        seed_means_a = np.array(seed_means_a)
        seed_means_b = np.array(seed_means_b)
        all_eps_a = np.array(all_eps_a)
        all_eps_b = np.array(all_eps_b)

        print(f"\n  Seed-level means:")
        print(f"    {args.algo_a:25s}: {seed_means_a}  (mean={np.mean(seed_means_a):.1f}, std={np.std(seed_means_a, ddof=1):.1f})")
        print(f"    {args.algo_b:25s}: {seed_means_b}  (mean={np.mean(seed_means_b):.1f}, std={np.std(seed_means_b, ddof=1):.1f})")

        # 1. Mann-Whitney U (seed-level)
        if len(seed_means_a) >= 2 and len(seed_means_b) >= 2:
            U, p_mw = stats.mannwhitneyu(seed_means_a, seed_means_b, alternative="two-sided")
            print(f"\n  1. Mann-Whitney U (seed means, two-sided):")
            print(f"     U={U:.1f}, p={p_mw:.4f}")
        else:
            print(f"\n  1. Mann-Whitney U: insufficient samples")

        # 2. Exact permutation test (seed-level)
        diff, p_perm, n_perms = permutation_test_mc(seed_means_a, seed_means_b)
        print(f"\n  2. Exact permutation test (seed means, two-sided):")
        print(f"     observed diff={diff:.1f}, p={p_perm:.4f} ({n_perms} permutations)")

        # 3. Welch's t-test (all episodes, liberal)
        t_stat, p_welch = stats.ttest_ind(all_eps_a, all_eps_b, equal_var=False)
        print(f"\n  3. Welch's t-test (all episodes, n={len(all_eps_a)} vs {len(all_eps_b)}):")
        print(f"     t={t_stat:.3f}, p={p_welch:.4f}")
        print(f"     NOTE: violates independence (episodes nested within seeds)")

        # 4. Effect size
        d = cohens_d(seed_means_a, seed_means_b)
        magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"\n  4. Cohen's d (seed means): {d:.3f} ({magnitude})")

        # 5. Normalized summary
        print(f"\n  Normalized fitness (/ {ref}):")
        print(f"    {args.algo_a:25s}: {np.mean(seed_means_a)/ref:.4f} +/- {np.std(seed_means_a, ddof=1)/ref:.4f}")
        print(f"    {args.algo_b:25s}: {np.mean(seed_means_b)/ref:.4f} +/- {np.std(seed_means_b, ddof=1)/ref:.4f}")
        print()


if __name__ == "__main__":
    main()
