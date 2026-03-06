import argparse
import numpy as np
import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec
from tensorneat.common import ACT, AGG

# ── Task presets ──────────────────────────────────────────────
# Each preset: list of (env_name, obs_size, act_size) tuples.
# Add new combinations here.
#  # All tasks as mp4
#   uv run python examples/brax/multi_task_visualize.py best_genome.npz                                 
                                                            
#   # Only hopper as gif
#   uv run python examples/brax/multi_task_visualize.py best_genome.npz --tasks hopper -f gif

#   # Custom output dir
#   uv run python examples/brax/multi_task_visualize.py best_genome.npz -d outputs/
PRESETS = {
    "hopper_walker": [
        ("hopper", 11, 3),
        ("walker2d", 17, 6),
    ],
}


def build_pipeline(preset_name):
    """Build a Pipeline for the given preset."""
    specs = PRESETS[preset_name]
    tasks = [
        TaskSpec(env=BraxEnv(env_name=name, max_step=1000),
                 obs_size=obs, act_size=act)
        for name, obs, act in specs
    ]
    max_obs = max(t.obs_size for t in tasks)
    max_act = max(t.act_size for t in tasks)

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                max_nodes=50,
                max_conns=200,
                num_inputs=max_obs,
                num_outputs=max_act,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=MultiTaskBraxEnv(tasks=tasks),
        seed=42,
        generation_limit=1,
        fitness_target=float("inf"),
    )
    return pipeline, tasks


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved multi-task genome")
    parser.add_argument("genome_path", help="Path to saved .npz genome file")
    parser.add_argument(
        "--preset", default="hopper_walker",
        choices=list(PRESETS.keys()),
        help="Task combination preset (default: hopper_walker)")
    parser.add_argument("--output-dir", "-d", default=".",
                        help="Directory for output files")
    parser.add_argument(
        "--format", "-f", default="mp4",
        choices=["rgb_array", "gif", "mp4"],
        help="Output format (default: mp4)")
    parser.add_argument(
        "--tasks", "-t", nargs="*", default=None,
        help="Subset of task env names to visualize (default: all)")
    args = parser.parse_args()

    data = np.load(args.genome_path)
    best = (jnp.array(data["nodes"]), jnp.array(data["conns"]))
    if "fitness" in data:
        print(f"Loaded genome with fitness: {data['fitness']}")

    pipeline, tasks = build_pipeline(args.preset)
    state = pipeline.setup()

    task_names = [name for name, _, _ in PRESETS[args.preset]]
    if args.tasks:
        indices = [task_names.index(t) for t in args.tasks]
    else:
        indices = list(range(len(tasks)))

    for idx in indices:
        env_name = task_names[idx]
        print(f"Visualizing task: {env_name}")
        if args.format == "rgb_array":
            frames = pipeline.show(state, best, task_index=idx,
                                   output_type="rgb_array")
            print(f"  Got {len(frames)} frames")
        else:
            save_path = f"{args.output_dir}/{env_name}.{args.format}"
            pipeline.show(state, best, task_index=idx,
                          output_type=args.format, save_path=save_path)
            print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
