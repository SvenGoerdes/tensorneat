import argparse
import numpy as np
import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG


def main():
    parser = argparse.ArgumentParser(description="Visualize a saved Hopper genome")
    parser.add_argument("genome_path", help="Path to saved .npz genome file")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument(
        "--format", "-f", default="mp4", choices=["rgb_array", "gif", "mp4"],
        help="Output format (default: mp4)",
    )
    args = parser.parse_args()

    # Load saved genome
    data = np.load(args.genome_path)
    best = (jnp.array(data["nodes"]), jnp.array(data["conns"]))
    if "fitness" in data:
        print(f"Loaded genome with fitness: {data['fitness']}")

    # Recreate the same pipeline config as hopper.py
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                num_inputs=11,
                num_outputs=3,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=BraxEnv(
            env_name="hopper",
            max_step=1000,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=5000,
    )

    state = pipeline.setup()

    save_path = args.output
    if args.format == "rgb_array":
        frames = pipeline.show(state, best, output_type="rgb_array")
        print(f"Got {len(frames)} frames as numpy array")
    else:
        if save_path is None:
            save_path = f"hopper.{args.format}"
        pipeline.show(state, best, output_type=args.format, save_path=save_path)


if __name__ == "__main__":
    main()
