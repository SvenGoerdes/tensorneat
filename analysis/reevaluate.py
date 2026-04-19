"""Re-evaluate genome .npz files using the existing eval pipeline."""

import json
import os
import sys
from datetime import datetime

import jax.numpy as jnp
import numpy as np

# Ensure project root is on path so tensorneat imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.evaluate_genomes import build_eval_pipeline, evaluate_genome, TASKS_CONFIG
from tensorneat.problem.rl import BRAX_REFERENCE_REWARDS

from analysis.merge import GenomeEntry

TASKS = [tc["env_name"] for tc in TASKS_CONFIG]


def run_reevaluation(
    entries: list[GenomeEntry],
    n_eval: int,
    backend: str = "mjx",
) -> list[dict]:
    """Evaluate all genome entries. Returns a list of run dicts matching the eval JSON schema."""
    print(f"Building NEAT pipeline (backend={backend})...")
    pipeline_neat, state_neat, tasks = build_eval_pipeline(is_ha_neat=False, backend=backend)
    print("Building HA-NEAT pipeline...")
    pipeline_haneat, state_haneat, _ = build_eval_pipeline(is_ha_neat=True, backend=backend)

    all_results = []
    for i, entry in enumerate(entries):
        is_ha = entry.algorithm == "ha_neat"
        pipeline = pipeline_haneat if is_ha else pipeline_neat
        state = state_haneat if is_ha else state_neat

        print(f"  [{i+1}/{len(entries)}] {entry.fname} ({entry.algorithm}, exp={entry.experiment})")
        data = np.load(entry.path)
        nodes = jnp.array(data["nodes"])
        conns = jnp.array(data["conns"])
        training_fitness = float(data["fitness"]) if "fitness" in data else None

        try:
            res = evaluate_genome(pipeline, state, tasks, nodes, conns, n_eval=n_eval)
            normalized = {
                env: res[env]["mean"] / BRAX_REFERENCE_REWARDS[env]
                for env in res
                if env in BRAX_REFERENCE_REWARDS
            }
            all_results.append({
                "file": entry.fname,
                "algorithm": entry.algorithm,
                "experiment": entry.experiment,
                "seed": entry.seed,
                "compat": entry.compat,
                "training_fitness": training_fitness,
                "tasks": res,
                "normalized": normalized,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({
                "file": entry.fname,
                "algorithm": entry.algorithm,
                "experiment": entry.experiment,
                "error": str(e),
            })

    return all_results


def build_combined_json(
    all_results: list[dict],
    n_eval: int,
    backend: str,
    experiments: list[str],
) -> dict:
    """Build a combined JSON dict in the same schema as evaluate_genomes.py output."""
    runs_ok = [r for r in all_results if "error" not in r]

    summary: dict[str, dict] = {}
    for algo in ["neat", "ha_neat"]:
        algo_runs = [r for r in runs_ok if r.get("algorithm") == algo]
        if not algo_runs:
            continue
        summary[algo] = {}
        for env in TASKS:
            means = [r["tasks"][env]["mean"] for r in algo_runs if env in r.get("tasks", {})]
            norms = [r["normalized"][env] for r in algo_runs if env in r.get("normalized", {})]
            if not means:
                continue
            summary[algo][env] = {
                "mean_across_seeds": float(np.mean(means)),
                "std_across_seeds": float(np.std(means)),
                "mean_normalized": float(np.mean(norms)),
                "std_normalized": float(np.std(norms)),
                "n_seeds": len(means),
            }

    return {
        "metadata": {
            "experiments": experiments,
            "backend": backend,
            "n_eval": n_eval,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "n_genomes": len(all_results),
            "experiment": "combined_final",
        },
        "runs": all_results,
        "summary": summary,
    }


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")
