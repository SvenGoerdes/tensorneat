"""
Batch visualize all genomes in a results folder.

Usage:
    uv run python eval/visualize_folder.py results/multi_task_neat_vs_haneat_v4
    uv run python eval/visualize_folder.py results/my_experiment/ --backend generalized
    uv run python eval/visualize_folder.py results/my_experiment/ --tasks hopper
    uv run python eval/visualize_folder.py results/my_experiment/ --format gif

Output:
    videos/<experiment_name>/<genome_stem>_<task>.mp4
    e.g. videos/my_experiment/neat_pop1024_gen500_seed42_walker2d.mp4

Assumption: all genomes in a folder share max_nodes=50, max_conns=200.
Two pipelines are built per run (one NEAT, one HA-NEAT) and reused across
all genomes of the same type to avoid repeated JAX JIT compilation.
"""

import argparse
import os
import glob
import numpy as np
import jax.numpy as jnp

# Resolve project root so output paths are stable regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, OriginConn, HANEATMutation
from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec
from tensorneat.common import ACT, AGG

# ── Task presets ──────────────────────────────────────────────
PRESETS = {
    "hopper_walker": [
        ("hopper", 11, 3),
        ("walker2d", 17, 6),
    ],
}


def build_pipeline(preset_name, is_ha_neat=False, backend="mjx"):
    """Build and return (pipeline, state, task_names)."""
    specs = PRESETS[preset_name]
    tasks = [
        TaskSpec(env=BraxEnv(env_name=name, max_step=1000, backend=backend),
                 obs_size=obs, act_size=act)
        for name, obs, act in specs
    ]
    max_obs = max(t.obs_size for t in tasks)
    max_act = max(t.act_size for t in tasks)

    if is_ha_neat:
        activation_fns = [ACT.tanh, ACT.sigmoid, ACT.relu, ACT.sin, ACT.identity]
        genome = DefaultGenome(
            max_nodes=50, max_conns=200,
            num_inputs=max_obs, num_outputs=max_act,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=activation_fns,
                aggregation_options=AGG.sum,
                activation_replace_rate=0.0,
            ),
            conn_gene=OriginConn(),
            mutation=HANEATMutation(activation_mutate_rate=0.1, max_conns=200),
            output_transform=ACT.tanh,
        )
    else:
        genome = DefaultGenome(
            max_nodes=50, max_conns=200,
            num_inputs=max_obs, num_outputs=max_act,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=genome,
        ),
        problem=MultiTaskBraxEnv(tasks=tasks),
        seed=42,
        generation_limit=1,
        fitness_target=float("inf"),
    )
    state = pipeline.setup()
    task_names = [name for name, _, _ in specs]
    return pipeline, state, task_names


def main():
    parser = argparse.ArgumentParser(
        description="Batch visualize all genomes in a results folder")
    parser.add_argument(
        "folder", help="Path to folder containing .npz genome files")
    parser.add_argument(
        "--preset", default="hopper_walker",
        choices=list(PRESETS.keys()),
        help="Task combination preset (default: hopper_walker)")
    parser.add_argument(
        "--format", "-f", default="mp4",
        choices=["gif", "mp4"],
        help="Output format (default: mp4)")
    parser.add_argument(
        "--tasks", "-t", nargs="*", default=None,
        help="Subset of task env names to visualize (default: all)")
    parser.add_argument(
        "--backend", "-b", default="mjx",
        choices=["mjx", "generalized", "positional", "spring"],
        help="Brax physics backend (default: mjx)")
    args = parser.parse_args()

    folder = args.folder.rstrip("/")
    if not os.path.isabs(folder):
        folder = os.path.join(PROJECT_ROOT, folder)
    experiment_name = os.path.basename(folder)
    output_base = os.path.join(PROJECT_ROOT, "videos", experiment_name)
    os.makedirs(output_base, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {folder}")
        return

    print(f"Found {len(npz_files)} genome(s) in '{folder}'")
    print(f"Output dir: {output_base}/")
    print()

    # Build pipelines lazily — only compile what's needed
    pipelines = {}  # "neat" | "ha_neat" -> (pipeline, state, task_names)

    for genome_path in npz_files:
        stem = os.path.splitext(os.path.basename(genome_path))[0]
        is_ha_neat = stem.startswith("ha_neat")
        algo_key = "ha_neat" if is_ha_neat else "neat"

        # Build pipeline on first encounter of this algorithm type
        if algo_key not in pipelines:
            algo_label = "HA-NEAT" if is_ha_neat else "NEAT"
            print(f"[setup] Compiling pipeline for {algo_label} (one-time JIT)...")
            pipelines[algo_key] = build_pipeline(
                args.preset, is_ha_neat=is_ha_neat, backend=args.backend)
            print(f"[setup] Done.\n")

        pipeline, state, task_names = pipelines[algo_key]

        data = np.load(genome_path)
        best = (jnp.array(data["nodes"]), jnp.array(data["conns"]))
        fitness_str = f"  fitness={data['fitness']}" if "fitness" in data else ""
        print(f"[genome] {stem}{fitness_str}")

        if args.tasks:
            indices = [task_names.index(t) for t in args.tasks]
        else:
            indices = list(range(len(task_names)))

        for idx in indices:
            env_name = task_names[idx]
            save_path = os.path.join(output_base, f"{stem}_{env_name}.{args.format}")
            print(f"  -> {env_name}: {save_path}")
            pipeline.show(state, best, task_index=idx,
                          output_type=args.format, save_path=save_path)

        print()

    print("Done.")


if __name__ == "__main__":
    main()
