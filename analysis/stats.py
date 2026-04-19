"""Statistical tests for NEAT vs HA-NEAT comparison. Writes results to files."""

import io
import json
import os

import numpy as np
from scipy import stats as scipy_stats

REFERENCE_REWARDS = {"hopper": 3000, "walker2d": 5000, "ant": 6000, "halfcheetah": 8000, "humanoid": 6000, "reacher": 5}
TASKS = ["hopper", "walker2d"]


def _load_helpers():
    """Lazy import of statistical_test helpers (avoids top-level path issues)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "text", "figures"))
    from statistical_test import permutation_test_exact, cohens_d
    return permutation_test_exact, cohens_d


def _seed_means(runs: list[dict], task: str) -> np.ndarray:
    return np.array([r["tasks"][task]["mean"] for r in runs if task in r.get("tasks", {})])


def _all_episodes(runs: list[dict], task: str) -> np.ndarray:
    eps = []
    for r in runs:
        if task in r.get("tasks", {}):
            eps.extend(r["tasks"][task]["episodes"])
    return np.array(eps)


def compute_normalized_min(runs: list[dict]) -> list[float]:
    """Compute normalized_min = min(norm_hopper, norm_walker2d) per seed."""
    out = []
    for r in runs:
        norms = r.get("normalized", {})
        if "hopper" in norms and "walker2d" in norms:
            out.append(min(norms["hopper"], norms["walker2d"]))
    return out


def compute_tests(runs_a: list[dict], runs_b: list[dict], label_a: str = "neat", label_b: str = "ha_neat") -> dict:
    """Run all statistical tests per task + normalized_min aggregate. Returns structured dict."""
    permutation_test_exact, cohens_d = _load_helpers()
    results = {}

    task_groups = {t: (_seed_means(runs_a, t), _seed_means(runs_b, t)) for t in TASKS}
    norm_min_a = np.array(compute_normalized_min(runs_a))
    norm_min_b = np.array(compute_normalized_min(runs_b))
    task_groups["normalized_min"] = (norm_min_a, norm_min_b)

    for task, (ma, mb) in task_groups.items():
        ref = REFERENCE_REWARDS.get(task, 1.0)

        entry: dict = {
            "n_a": int(len(ma)),
            "n_b": int(len(mb)),
            "mean_a": float(np.mean(ma)) if len(ma) else float("nan"),
            "mean_b": float(np.mean(mb)) if len(mb) else float("nan"),
            "std_a": float(np.std(ma, ddof=1)) if len(ma) > 1 else float("nan"),
            "std_b": float(np.std(mb, ddof=1)) if len(mb) > 1 else float("nan"),
            "sem_a": float(np.std(ma, ddof=1) / np.sqrt(len(ma))) if len(ma) > 1 else float("nan"),
            "sem_b": float(np.std(mb, ddof=1) / np.sqrt(len(mb))) if len(mb) > 1 else float("nan"),
            "mean_norm_a": float(np.mean(ma) / ref) if len(ma) else float("nan"),
            "mean_norm_b": float(np.mean(mb) / ref) if len(mb) else float("nan"),
        }

        if len(ma) >= 2 and len(mb) >= 2:
            U, p_mw = scipy_stats.mannwhitneyu(ma, mb, alternative="two-sided")
            entry["mannwhitney_u"] = float(U)
            entry["p_mannwhitney"] = float(p_mw)
        else:
            entry["mannwhitney_u"] = None
            entry["p_mannwhitney"] = None

        diff, p_perm, n_perms = permutation_test_exact(ma, mb)
        entry["observed_diff"] = float(diff)
        entry["p_permutation"] = float(p_perm)
        entry["n_permutations"] = int(n_perms)

        d = cohens_d(ma, mb)
        mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        entry["cohens_d"] = float(d)
        entry["magnitude"] = mag

        results[task] = entry

    return results


def format_txt_report(test_results: dict, label_a: str, label_b: str) -> str:
    buf = io.StringIO()

    def pr(*args):
        print(*args, file=buf)

    pr(f"Statistical Comparison: {label_a} vs {label_b}")
    pr("=" * 70)

    for task, r in test_results.items():
        ref = REFERENCE_REWARDS.get(task, 1.0)
        pr(f"\n{'='*70}")
        pr(f"  Task: {task.upper()}")
        pr(f"{'='*70}")
        pr(f"\n  n: {label_a}={r['n_a']}, {label_b}={r['n_b']}")
        pr(f"  Mean: {label_a}={r['mean_a']:.1f} ± {r.get('std_a', float('nan')):.1f}  "
           f"{label_b}={r['mean_b']:.1f} ± {r.get('std_b', float('nan')):.1f}")
        pr(f"  Norm mean (/{ref}): {label_a}={r['mean_norm_a']:.4f}  {label_b}={r['mean_norm_b']:.4f}")

        if r["p_mannwhitney"] is not None:
            pr(f"\n  1. Mann-Whitney U: U={r['mannwhitney_u']:.1f}, p={r['p_mannwhitney']:.4f}")
        else:
            pr("\n  1. Mann-Whitney U: insufficient samples")

        pr(f"\n  2. Exact permutation test:")
        pr(f"     observed diff={r['observed_diff']:.1f}, p={r['p_permutation']:.4f} ({r['n_permutations']} permutations)")

        pr(f"\n  3. Cohen's d: {r['cohens_d']:.3f} ({r['magnitude']})")

    return buf.getvalue()


def write_stats(test_results: dict, output_dir: str, label_a: str, label_b: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "per_task_tests.txt")
    with open(txt_path, "w") as f:
        f.write(format_txt_report(test_results, label_a, label_b))
    print(f"  Saved: {txt_path}")

    json_path = os.path.join(output_dir, "per_task_tests.json")
    with open(json_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"  Saved: {json_path}")
