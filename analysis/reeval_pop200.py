"""Re-evaluate all pop200 genomes (3 conditions, n=30 each) on MJX — vmapped.

Unlike eval/evaluate_genomes.py (which loops over episodes in Python, leaving
the GPU idle between 1-episode calls), this batches all n_eval episodes per task
into a single jax.vmap call over _evaluate_once. The GPU stays saturated, so the
full 90-genome run finishes in minutes instead of hours.

Reuses analysis.merge.collect_genomes for the genome inventory and
eval.evaluate_genomes.build_eval_pipeline for the (NEAT / HA-NEAT) pipelines.
Writes a combined.json matching the standard eval schema, consumed by
analysis/report_pop200.py for the 3-way statistical comparison.

Output is line-buffered and flushed per genome so progress is visible in the log.

Usage:
    python -m analysis.reeval_pop200 --n-eval 20
    python -m analysis.reeval_pop200 --n-eval 3   # smoke test
"""

import argparse
import functools
import os
import sys
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.merge import collect_genomes
from analysis.reevaluate import build_combined_json, save_json
from eval.evaluate_genomes import build_eval_pipeline
from tensorneat.problem.rl import BRAX_REFERENCE_REWARDS

EXPERIMENT = "multi_task_neat_vs_haneat_pop200"
BACKEND = "mjx"


def _print(*a):
    print(*a, flush=True)


def make_vmapped_task_evaluators(pipeline, state, tasks, n_eval: int):
    """Build one jitted, vmapped (over n_eval episodes) evaluator per task.

    Returns a list of callables: eval_fn(transformed_params, base_key) -> (n_eval,) rewards.
    """
    evaluators = []
    for i, task in enumerate(tasks):
        adapted = pipeline.problem._make_adapted_act_func(
            pipeline.algorithm.forward, task.obs_size, task.act_size
        )
        env = task.env

        def single(key, params, _adapted=adapted, _env=env):
            # _evaluate_once with record_episode=False returns a scalar total reward.
            return _env._evaluate_once(
                state, key, _adapted, params, _env.action_policy, False, _env.obs_normalization
            )

        @functools.partial(jax.jit, static_argnums=())
        def batch(params, base_key, _single=single, _i=i):
            keys = jax.random.split(jax.random.fold_in(base_key, _i), n_eval)
            return jax.vmap(_single, in_axes=(0, None))(keys, params)

        evaluators.append((task.env.env_name, batch))
    return evaluators


def evaluate_genome_vmap(pipeline, state, evaluators, nodes, conns, n_eval: int) -> dict:
    transformed = pipeline.algorithm.transform(state, (nodes, conns))
    base_key = jax.random.PRNGKey(0)
    results = {}
    for env_name, batch in evaluators:
        rewards = np.asarray(batch(transformed, base_key))  # (n_eval,)
        results[env_name] = {
            "episodes": [float(x) for x in rewards],
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }
    return results


def run(entries, n_eval: int, backend: str) -> list[dict]:
    _print(f"Building NEAT pipeline (backend={backend})...")
    pipe_neat, state_neat, tasks = build_eval_pipeline(is_ha_neat=False, backend=backend)
    _print("Building HA-NEAT pipeline...")
    pipe_ha, state_ha, _ = build_eval_pipeline(is_ha_neat=True, backend=backend)

    ev_neat = make_vmapped_task_evaluators(pipe_neat, state_neat, tasks, n_eval)
    ev_ha = make_vmapped_task_evaluators(pipe_ha, state_ha, tasks, n_eval)

    out = []
    n = len(entries)
    for idx, entry in enumerate(entries):
        is_ha = entry.algorithm in ("ha_neat", "ha_neat_ablation")
        pipe = pipe_ha if is_ha else pipe_neat
        st = state_ha if is_ha else state_neat
        ev = ev_neat if not is_ha else ev_ha

        _print(f"  [{idx+1}/{n}] {entry.fname} ({entry.algorithm})")
        data = np.load(entry.path)
        nodes = jnp.array(data["nodes"])
        conns = jnp.array(data["conns"])
        training_fitness = float(data["fitness"]) if "fitness" in data else None

        try:
            res = evaluate_genome_vmap(pipe, st, ev, nodes, conns, n_eval)
            normalized = {
                env: res[env]["mean"] / BRAX_REFERENCE_REWARDS[env]
                for env in res if env in BRAX_REFERENCE_REWARDS
            }
            out.append({
                "file": entry.fname, "algorithm": entry.algorithm,
                "experiment": entry.experiment, "seed": entry.seed, "compat": entry.compat,
                "training_fitness": training_fitness, "tasks": res, "normalized": normalized,
            })
        except Exception as e:
            _print(f"    ERROR: {e}")
            out.append({"file": entry.fname, "algorithm": entry.algorithm,
                        "experiment": entry.experiment, "error": str(e)})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-evaluate pop200 genomes (3-way, vmapped).")
    p.add_argument("--n-eval", type=int, default=20, help="Episodes per genome per task")
    p.add_argument("--out", default=None, help="Output combined.json path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = os.path.join(PROJECT_ROOT, "results")

    entries = collect_genomes(results_root, [EXPERIMENT])
    if not entries:
        raise SystemExit(f"No genomes found for {EXPERIMENT} under {results_root}")

    counts = Counter(e.algorithm for e in entries)
    _print(f"Collected {len(entries)} genomes from {EXPERIMENT}:")
    for algo in ("neat", "ha_neat", "ha_neat_ablation"):
        _print(f"  {algo:18s}: {counts.get(algo, 0)}")
    _print(f"Re-evaluating {args.n_eval} episodes/task on {BACKEND} (vmapped)...\n")

    results = run(entries, n_eval=args.n_eval, backend=BACKEND)

    combined = build_combined_json(results, n_eval=args.n_eval, backend=BACKEND, experiments=[EXPERIMENT])
    combined["metadata"]["experiment"] = EXPERIMENT

    out = args.out or os.path.join(PROJECT_ROOT, "analysis", "outputs", "pop200", "combined.json")
    save_json(combined, out)

    n_ok = sum(1 for r in results if "error" not in r)
    _print(f"\nDone: {n_ok} ok, {len(results) - n_ok} errors. -> {out}")


if __name__ == "__main__":
    main()
