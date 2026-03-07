"""
Compare standard NEAT vs HA-NEAT on the Hopper Brax environment.
Both runs are tracked via MLflow under the same experiment.
"""

import mlflow

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, OriginConn, HANEATMutation
from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

SEED = 42
POP_SIZE = 1000
GENERATION_LIMIT = 200
FITNESS_TARGET = 5000
MAX_CONNS = 100
EXPERIMENT_NAME = "NEAT_vs_HA-NEAT_Hopper"


def make_neat_pipeline():
    """Standard NEAT — single activation (tanh), no historical markers."""
    return Pipeline(
        algorithm=NEAT(
            pop_size=POP_SIZE,
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
        problem=BraxEnv(env_name="hopper", max_step=1000),
        seed=SEED,
        generation_limit=GENERATION_LIMIT,
        fitness_target=FITNESS_TARGET,
        mlflow_tracking=True,
        mlflow_run_name="NEAT",
    )


def make_ha_neat_pipeline():
    """HA-NEAT — heterogeneous activations with speciation protection."""
    return Pipeline(
        algorithm=NEAT(
            pop_size=POP_SIZE,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                num_inputs=11,
                num_outputs=3,
                max_conns=MAX_CONNS,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.sigmoid, ACT.relu, ACT.sin, ACT.identity],
                    aggregation_options=AGG.sum,
                    activation_replace_rate=0.0,
                ),
                conn_gene=OriginConn(),
                mutation=HANEATMutation(
                    activation_mutate_rate=0.1,
                    max_conns=MAX_CONNS,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=BraxEnv(env_name="hopper", max_step=1000),
        seed=SEED,
        generation_limit=GENERATION_LIMIT,
        fitness_target=FITNESS_TARGET,
        mlflow_tracking=True,
        mlflow_run_name="HA-NEAT",
    )


def run_experiment(pipeline, name):
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)

    print(f"[{name}] Best fitness: {pipeline.best_fitness:.2f}\n")
    return pipeline.best_fitness


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    neat_best = run_experiment(make_neat_pipeline(), "NEAT")
    ha_neat_best = run_experiment(make_ha_neat_pipeline(), "HA-NEAT")

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  NEAT:    best={neat_best:8.2f}")
    print(f"  HA-NEAT: best={ha_neat_best:8.2f}")
    print(f"{'='*60}")
    print(f"\n  View results: mlflow ui")
