"""Evaluate all saved genomes in a results directory on each task independently."""

import os
import glob
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, OriginConn, HANEATMutation
from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec, BRAX_REFERENCE_REWARDS
from tensorneat.common import ACT, AGG, State


TASKS_CONFIG = [
    {"env_name": "hopper",   "max_step": 1000, "obs_size": 11, "act_size": 3, "weight": 1.0},
    {"env_name": "walker2d", "max_step": 1000, "obs_size": 17, "act_size": 6, "weight": 1.0},
]
MAX_NODES = 50
MAX_CONNS = 200
N_EVAL = 5  # number of episodes per task to average over


def build_eval_pipeline(is_ha_neat: bool, backend: str = "mjx"):
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

    algorithm = NEAT(genome=genome, pop_size=2)  # minimal pop, we only need transform/forward
    problem = MultiTaskBraxEnv(tasks=tasks)

    pipeline = Pipeline(algorithm=algorithm, problem=problem, seed=0, generation_limit=1)
    state = pipeline.setup()
    return pipeline, state, tasks


def evaluate_genome(pipeline, state, tasks, nodes, conns, n_eval=N_EVAL):
    transformed = pipeline.algorithm.transform(state, (nodes, conns))

    results = {}
    for i, task in enumerate(tasks):
        adapted = pipeline.problem._make_adapted_act_func(
            pipeline.algorithm.forward, task.obs_size, task.act_size
        )
        fitnesses = []
        for ep in range(n_eval):
            key = jax.random.PRNGKey(ep * 100 + i)
            f = task.env.evaluate(state, key, adapted, transformed)
            fitnesses.append(float(f))
        mean_f = np.mean(fitnesses)
        results[task.env.env_name] = mean_f
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results/multi_task_neat_vs_haneat")
    parser.add_argument("--n_eval", type=int, default=N_EVAL)
    parser.add_argument("--backend", default="mjx", choices=["mjx", "generalized", "positional", "spring"])
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.results_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.results_dir}")
        return

    print(f"Found {len(npz_files)} genome files. Evaluating each on {args.n_eval} episodes per task.\n")
    print(f"Backend: {args.backend}")
    print(f"{'File':<55} {'hopper':>10} {'walker2d':>10} {'hop_norm':>10} {'wal_norm':>10}")
    print("-" * 100)

    # Build pipelines once each
    pipeline_neat,    state_neat,    tasks = build_eval_pipeline(is_ha_neat=False, backend=args.backend)
    pipeline_haneat,  state_haneat,  _     = build_eval_pipeline(is_ha_neat=True,  backend=args.backend)

    for path in npz_files:
        fname = os.path.basename(path)
        is_ha = fname.startswith("ha_neat")
        pipeline = pipeline_haneat if is_ha else pipeline_neat
        state    = state_haneat    if is_ha else state_neat

        data  = np.load(path)
        nodes = jnp.array(data["nodes"])
        conns = jnp.array(data["conns"])

        try:
            res = evaluate_genome(pipeline, state, tasks, nodes, conns, n_eval=args.n_eval)
            hopper   = res.get("hopper",   float("nan"))
            walker2d = res.get("walker2d", float("nan"))
            hop_norm = hopper   / BRAX_REFERENCE_REWARDS["hopper"]
            wal_norm = walker2d / BRAX_REFERENCE_REWARDS["walker2d"]
            print(f"{fname:<55} {hopper:>10.1f} {walker2d:>10.1f} {hop_norm:>10.3f} {wal_norm:>10.3f}")
        except Exception as e:
            print(f"{fname:<55} ERROR: {e}")

    print("-" * 100)


if __name__ == "__main__":
    main()
