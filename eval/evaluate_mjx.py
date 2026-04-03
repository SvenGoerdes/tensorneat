"""Evaluate and visualize a saved genome using the MJX backend."""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp

from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, OriginConn, HANEATMutation
from tensorneat.pipeline import Pipeline
from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec, BRAX_REFERENCE_REWARDS
from tensorneat.common import ACT, AGG

TASKS_CONFIG = [
    {"env_name": "hopper",   "max_step": 1000, "obs_size": 11, "act_size": 3, "weight": 1.0},
    {"env_name": "walker2d", "max_step": 1000, "obs_size": 17, "act_size": 6, "weight": 1.0},
]
MAX_NODES = 50
MAX_CONNS = 200


def build_pipeline(is_ha_neat: bool, backend: str):
    tasks = []
    for tc in TASKS_CONFIG:
        env = BraxEnv(env_name=tc["env_name"], max_step=tc["max_step"], backend=backend)
        tasks.append(TaskSpec(
            env=env,
            obs_size=tc["obs_size"],
            act_size=tc["act_size"],
            weight=tc["weight"],
            max_reward=BRAX_REFERENCE_REWARDS.get(tc["env_name"]),
        ))

    num_inputs  = max(t.obs_size for t in tasks)
    num_outputs = max(t.act_size for t in tasks)

    if is_ha_neat:
        activation_fns = [ACT.tanh, ACT.sigmoid, ACT.relu, ACT.sin, ACT.identity]
        genome = DefaultGenome(
            max_nodes=MAX_NODES, max_conns=MAX_CONNS,
            num_inputs=num_inputs, num_outputs=num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(activation_options=activation_fns, aggregation_options=AGG.sum, activation_replace_rate=0.0),
            conn_gene=OriginConn(),
            mutation=HANEATMutation(activation_mutate_rate=0.1, max_conns=MAX_CONNS),
            output_transform=ACT.tanh,
        )
    else:
        genome = DefaultGenome(
            max_nodes=MAX_NODES, max_conns=MAX_CONNS,
            num_inputs=num_inputs, num_outputs=num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum),
            output_transform=ACT.tanh,
        )

    algorithm = NEAT(genome=genome, pop_size=2)
    problem = MultiTaskBraxEnv(tasks=tasks)
    pipeline = Pipeline(algorithm=algorithm, problem=problem, seed=0, generation_limit=1)
    state = pipeline.setup()
    return pipeline, state, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("genome_path", help="Path to .npz genome file")
    parser.add_argument("--backend", default="mjx", choices=["generalized", "mjx", "positional", "spring"])
    parser.add_argument("--tasks", nargs="*", default=["walker2d", "hopper"])
    parser.add_argument("--output-dir", "-d", default=".")
    parser.add_argument("--format", "-f", default="mp4", choices=["mp4", "gif", "rgb_array"])
    args = parser.parse_args()

    data = np.load(args.genome_path)
    nodes = jnp.array(data["nodes"])
    conns = jnp.array(data["conns"])
    print(f"Loaded genome with fitness: {data.get('fitness', 'N/A')}")

    is_ha_neat = os.path.basename(args.genome_path).startswith("ha_neat")
    print(f"Algorithm: {'HA-NEAT' if is_ha_neat else 'NEAT'}")
    print(f"Backend:   {args.backend}")

    pipeline, state, tasks = build_pipeline(is_ha_neat, backend=args.backend)
    best = (nodes, conns)
    transformed = pipeline.algorithm.transform(state, best)

    for task_name in args.tasks:
        task_idx = next((i for i, t in enumerate(TASKS_CONFIG) if t["env_name"] == task_name), None)
        if task_idx is None:
            print(f"Unknown task: {task_name}")
            continue

        t = pipeline.problem.tasks[task_idx]
        task = task_name
        adapted = pipeline.problem._make_adapted_act_func(
            pipeline.algorithm.forward, t.obs_size, t.act_size
        )
        key = jax.random.PRNGKey(42)
        fitness = float(t.env.evaluate(state, key, adapted, transformed))
        print(f"\n{task_name}: reward = {fitness:.1f} (norm: {fitness / BRAX_REFERENCE_REWARDS[task_name]:.3f})")

        save_path = os.path.join(args.output_dir, f"{task_name}_{args.backend}.{args.format}")
        print(f"Visualizing {task_name} → {save_path}")
        t.env.show(state, key, adapted, transformed, save_path=save_path, output_type=args.format)


if __name__ == "__main__":
    main()
