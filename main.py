"""Experiment orchestrator for multi-task NEAT vs HA-NEAT comparisons."""

import argparse
import itertools
import os
import traceback

import numpy as np
import jax
import yaml

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode, OriginConn, HANEATMutation
from tensorneat.genome.operations.distance import DefaultDistance
from tensorneat.problem.rl import BraxEnv, MultiTaskBraxEnv, TaskSpec, BRAX_REFERENCE_REWARDS
from tensorneat.common import ACT, AGG

STRUCTURAL_KEYS = {"tasks", "activation_options", "experiment_name", "mlflow_tracking", "per_task_tracking", "backend"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(config: dict) -> list[dict]:
    sweep_keys = []
    sweep_values = []
    fixed = {}

    for key, value in config.items():
        if isinstance(value, list) and key not in STRUCTURAL_KEYS:
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed[key] = value

    grid = []
    for combo in itertools.product(*sweep_values):
        run_config = dict(fixed)
        for key, val in zip(sweep_keys, combo):
            run_config[key] = val
        grid.append(run_config)

    return grid


def build_tasks(task_configs: list[dict], backend: str = "generalized") -> list[TaskSpec]:
    tasks = []
    for tc in task_configs:
        env = BraxEnv(env_name=tc["env_name"], max_step=tc["max_step"], backend=backend)
        tasks.append(
            TaskSpec(
                env=env,
                obs_size=tc["obs_size"],
                act_size=tc["act_size"],
                weight=tc["weight"],
                max_reward=BRAX_REFERENCE_REWARDS.get(tc["env_name"]),
            )
        )
    return tasks


def build_pipeline(run_config: dict) -> Pipeline:
    backend = run_config.get("backend", "mjx")
    tasks = build_tasks(run_config["tasks"], backend=backend)
    num_inputs = max(t.obs_size for t in tasks)
    num_outputs = max(t.act_size for t in tasks)

    algorithm_type = run_config["algorithm_type"]
    max_nodes = run_config["max_nodes"]
    max_conns = run_config["max_conns"]
    distance = DefaultDistance(
        compatibility_disjoint=run_config.get("compatibility_disjoint", 1.0),
    )

    if algorithm_type == "neat":
        genome = DefaultGenome(
            max_nodes=max_nodes,
            max_conns=max_conns,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            distance=distance,
            output_transform=ACT.tanh,
        )
    elif algorithm_type == "ha_neat":
        activation_fns = [getattr(ACT, name) for name in run_config["activation_options"]]
        genome = DefaultGenome(
            max_nodes=max_nodes,
            max_conns=max_conns,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=activation_fns,
                aggregation_options=AGG.sum,
                activation_replace_rate=0.0,
            ),
            conn_gene=OriginConn(),
            mutation=HANEATMutation(
                activation_mutate_rate=run_config["activation_mutate_rate"],
                max_conns=max_conns,
            ),
            distance=distance,
            output_transform=ACT.tanh,
        )
    elif algorithm_type == "ha_neat_ablation":
        # Ablation control: HA-NEAT machinery (OriginConn, HANEATMutation, marker reassignment)
        # but restricted to tanh only — activation diversity is removed.
        # If this ≈ neat, the only meaningful difference between neat and ha_neat is
        # activation diversity, not the underlying mutation/speciation machinery.
        genome = DefaultGenome(
            max_nodes=max_nodes,
            max_conns=max_conns,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
                activation_replace_rate=0.0,
            ),
            conn_gene=OriginConn(),
            mutation=HANEATMutation(
                activation_mutate_rate=run_config["activation_mutate_rate"],
                max_conns=max_conns,
            ),
            distance=distance,
            output_transform=ACT.tanh,
        )
    else:
        raise ValueError(f"Unknown algorithm_type: {algorithm_type}")

    agg_mode = run_config.get("aggregation_mode", "normalized_sum")
    run_name = (
        f"{algorithm_type}_{agg_mode}_pop{run_config['pop_size']}"
        f"_gen{run_config['generation_limit']}"
        f"_seed{run_config['seed']}"
    )

    algorithm = NEAT(
        genome=genome,
        pop_size=run_config["pop_size"],
        species_size=run_config["species_size"],
        survival_threshold=run_config["survival_threshold"],
        compatibility_threshold=run_config["compatibility_threshold"],
        max_stagnation=run_config.get("max_stagnation", 15),
        species_number_calculate_by=run_config.get("species_number_calculate_by", "rank"),
    )

    return Pipeline(
        algorithm=algorithm,
        problem=MultiTaskBraxEnv(
            tasks=tasks,
            aggregation_mode=run_config.get("aggregation_mode", "normalized_sum"),
        ),
        seed=run_config["seed"],
        generation_limit=run_config["generation_limit"],
        fitness_target=run_config["fitness_target"],
        mlflow_tracking=run_config.get("mlflow_tracking", False),
        mlflow_run_name=run_name,
        mlflow_experiment_name=run_config.get("experiment_name"),
        mlflow_extra_params={
            "algorithm_type": algorithm_type,
            "aggregation_mode": run_config.get("aggregation_mode", "normalized_sum"),
            "backend": run_config.get("backend", "generalized"),
            "tasks": ",".join(t["env_name"] for t in run_config["tasks"]),
            "max_step": ",".join(str(t["max_step"]) for t in run_config["tasks"]),
            "species_size": run_config["species_size"],
            "survival_threshold": run_config["survival_threshold"],
            "compatibility_threshold": run_config["compatibility_threshold"],
            "compatibility_disjoint": run_config.get("compatibility_disjoint", 1.0),
            "max_stagnation": run_config.get("max_stagnation", 15),
            "species_number_calculate_by": run_config.get("species_number_calculate_by", "rank"),
            "max_nodes": run_config["max_nodes"],
            "max_conns": run_config["max_conns"],
            "activation_mutate_rate": run_config["activation_mutate_rate"],
            **({"activation_options": ",".join(run_config["activation_options"])}
               if algorithm_type == "ha_neat" else
               {"activation_options": "tanh"}
               if algorithm_type == "ha_neat_ablation" else {}),
        },
        per_task_tracking=run_config.get("per_task_tracking", True),
    )


def run_single(run_config: dict, results_dir: str) -> tuple[float, float]:
    agg_mode = run_config.get("aggregation_mode", "normalized_sum")
    run_name = (
        f"{run_config['algorithm_type']}_{agg_mode}_pop{run_config['pop_size']}"
        f"_gen{run_config['generation_limit']}"
        f"_gen{run_config['compatibility_disjoint']}"
        f"_seed{run_config['seed']}"
    )
    try:
        pipeline = build_pipeline(run_config)
        state = pipeline.setup()
        state, best = pipeline.auto_run(state)

        best_nodes, best_conns = jax.device_get(best)
        np.savez(
            os.path.join(results_dir, f"{run_name}.npz"),
            nodes=best_nodes,
            conns=best_conns,
            fitness=pipeline.best_fitness,
        )
        print(f"[{run_name}] Best fitness: {pipeline.best_fitness:.4f}")
        return pipeline.best_fitness, pipeline.best_fitness
    except Exception:
        traceback.print_exc()
        print(f"[{run_name}] FAILED")
        return float("nan"), float("nan")


def main():
    parser = argparse.ArgumentParser(description="Run NEAT/HA-NEAT experiment grid")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    grid = build_grid(config)

    # Identify sweep dimensions for summary
    sweep_dims = {
        k: v for k, v in config.items()
        if isinstance(v, list) and k not in STRUCTURAL_KEYS
    }

    print(f"\n{'='*60}")
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Total runs: {len(grid)}")
    print(f"  Sweep dimensions:")
    for k, v in sweep_dims.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    results_dir = os.path.join("results", config["experiment_name"])
    os.makedirs(results_dir, exist_ok=True)

    for i, run_config in enumerate(grid):
        print(f"\n--- Run {i+1}/{len(grid)} ---")
        print(f"  algorithm_type={run_config['algorithm_type']}, "
              f"pop_size={run_config['pop_size']}, "
              f"aggregation_mode={run_config['aggregation_mode']}, "
              f"generation_limit={run_config['generation_limit']}, "
              f"seed={run_config['seed']}")
        run_single(run_config, results_dir)

    print(f"\n{'='*60}")
    print(f"  All runs complete. Results saved to {results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
