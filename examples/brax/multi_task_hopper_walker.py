from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode

from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec
from tensorneat.common import ACT, AGG

if __name__ == "__main__":
    hopper = TaskSpec(
        env=BraxEnv(env_name="hopper", max_step=1000),
        obs_size=11,
        act_size=3,
        weight=1.0,
    )
    walker = TaskSpec(
        env=BraxEnv(env_name="walker2d", max_step=1000),
        obs_size=17,
        act_size=6,
        weight=1.0,
    )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                max_nodes=50,
                max_conns=200,
                num_inputs=17,
                num_outputs=6,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=MultiTaskBraxEnv(
            tasks=[hopper, walker],
        ),
        seed=42,
        generation_limit=100,
        fitness_target=5000,
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
