"""
Hyperparameter search for single-task Hopper (brax) using NEAT.

Sweeps over:
  pop_size:    [512, 1024, 2048]
  generations: [100, 200, 300]
  seeds:       [42, 123, 456]

Total runs: 3 × 3 × 3 = 27
Results are saved to a CSV file and per-run .npz files.
"""

import itertools
import csv
import os
import time
import numpy as np
import jax

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

# ---------- search grid ----------
POP_SIZES = [512, 1024, 2048]
GENERATIONS = [100, 200, 300]
SEEDS = [42, 123, 456]

RESULTS_DIR = "hyperparam_results"
CSV_FILE = os.path.join(RESULTS_DIR, "results.csv")


def run_single(pop_size: int, generation_limit: int, seed: int = 42):
    """Run one NEAT experiment and return the best fitness."""
    print(f"\n{'='*60}")
    print(f"  pop_size={pop_size}  generations={generation_limit}  seed={seed}")
    print(f"{'='*60}\n")

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=pop_size,
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
        seed=seed,
        generation_limit=generation_limit,
        fitness_target=5000,
        mlflow_tracking=True,
        mlflow_run_name=f"hopper_neat_pop{pop_size}_gen{generation_limit}_seed{seed}",
    )

    state = pipeline.setup()
    t0 = time.time()
    state, best = pipeline.auto_run(state)
    elapsed = time.time() - t0

    best_nodes, best_conns = jax.device_get(best)
    best_fitness = float(pipeline.best_fitness)

    # save per-run genome
    tag = f"pop{pop_size}_gen{generation_limit}_seed{seed}"
    npz_path = os.path.join(RESULTS_DIR, f"best_genome_{tag}.npz")
    np.savez(npz_path, nodes=best_nodes, conns=best_conns, fitness=best_fitness)

    print(f"  → fitness={best_fitness:.4f}  time={elapsed:.1f}s  saved={npz_path}")
    return best_fitness, elapsed


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # write CSV header
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pop_size", "generation_limit", "seed", "best_fitness", "time_s"])

    for pop_size, gen_limit, seed in itertools.product(POP_SIZES, GENERATIONS, SEEDS):
        fitness, elapsed = run_single(pop_size, gen_limit, seed=seed)

        # append result
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pop_size, gen_limit, seed, fitness, f"{elapsed:.1f}"])

    # print summary
    print(f"\n{'='*60}")
    print("  HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {CSV_FILE}\n")

    with open(CSV_FILE) as f:
        print(f.read())


if __name__ == "__main__":
    main()
