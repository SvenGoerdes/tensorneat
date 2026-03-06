# Update: Multi-Task NEAT for Simultaneous RL Evolution

**Date:** 2026-03-05
**Branch:** `dev/multi-task`
**Commit:** `650883f`

## What changed

Introduced multi-task support that allows a single shared NEAT network to be evolved on multiple RL environments simultaneously. No changes were needed to Pipeline, Algorithm, or Genome — the feature is fully self-contained.

### New files

- **`src/tensorneat/problem/rl/multi_task.py`** — Core implementation:
  - `TaskSpec`: dataclass to define a sub-task (environment, obs/act sizes, weight).
  - `MultiTaskBraxEnv`: evaluates one shared network across all tasks by zero-padding observations to the max obs size and slicing actions to each task's native action size.
  - `FitnessAggregator` / `WeightedSum`: pluggable strategy for combining per-task fitnesses into a single scalar (weighted sum by default).

- **`examples/brax/multi_task_hopper_walker.py`** — Example script evolving a single network on Hopper (obs=11, act=3) and Walker2D (obs=17, act=6) simultaneously with a population of 1000.

## Key design decisions

- **Zero-padding + slicing** keeps the genome fixed-size (max obs/act across tasks) while each task only sees/uses its relevant dimensions.
- **Weighted sum aggregation** is the default but can be swapped out via the `aggregator` parameter.
- **No upstream changes** — works entirely through the existing `BaseProblem` interface, so it plugs directly into the standard `Pipeline`.
