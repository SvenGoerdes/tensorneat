import numpy as np
import jax

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode

from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

if __name__ == "__main__":
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
        generation_limit=200,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)

    # save best genome
    best_nodes, best_conns = jax.device_get(best)
    np.savez(
        "best_genome_hopperv2.npz",
        nodes=best_nodes,
        conns=best_conns,
        fitness=pipeline.best_fitness,
    )
    print(f"Best genome saved (fitness: {pipeline.best_fitness:.4f})")
