# Feature Log

Running record of features implemented for the NEAT vs HA-NEAT thesis comparison.

---

## [2026-03-17] Experiment Orchestrator (`main.py`)

**Branch:** `feat/experiment-orchestrator`
**Status:** In progress

### What
Config-driven experiment grid runner that sweeps over algorithm types, population sizes, seeds, and generation limits via Cartesian product of list-valued YAML keys.

### Why
Needed a single entrypoint to run all NEAT vs HA-NEAT comparison conditions reproducibly without manually launching each run.

### Key files changed
- `main.py` — primary entrypoint with `build_grid()`, `build_pipeline()`, `run_single()`
- `config.yaml` — experiment config defining sweep dimensions

### Notes
Structural keys (`tasks`, `activation_options`, `experiment_name`, `mlflow_tracking`, `per_task_tracking`) are never swept. Results saved to `results/<experiment_name>/<run_name>.npz`.

---

## [2026-03-12] Normalized Fitness Aggregation + Per-Task Metric Logging

**Branch:** `feat/normalized-fitness-aggregation`
**Status:** Complete (merged to main)

### What
Added `NormalizedWeightedSum` aggregator that divides per-task fitness by `max_reward` before weighting. Auto-selected when any `TaskSpec.max_reward != 1.0`. Added `BRAX_REFERENCE_REWARDS` lookup dict. `per_task_evaluate()` now logs both raw and normalized fitness per task.

### Why
Hopper (max ~3000) and Walker2D (max ~5000) have different reward scales — plain `WeightedSum` would bias toward Walker2D. Normalization puts both tasks on a 0–1 scale for fair aggregation and comparable MLflow metrics.

### Key files changed
- `src/tensorneat/problem/rl/multi_task.py` — `NormalizedWeightedSum`, `BRAX_REFERENCE_REWARDS`, auto-selection logic, dual metric logging
- `test/test_multi_task_aggregation.py` — unit tests for aggregation and normalization

---

## [2026-03-10] HA-NEAT (Heterogeneous Activation NEAT)

**Branch:** `dev/ha-neat`
**Status:** Complete (merged to main)

### What
Implemented `HANEATMutation`, a subclass of `DefaultMutation` that gives each node its own activation function. Three mechanisms: random activation on node creation, one-node-per-generation controlled activation mutation with historical marker reassignment, and prevention of uncontrolled activation changes.

### Why
Core treatment condition of the thesis. HA-NEAT's activation diversity and speciation protection are hypothesized to improve multi-task performance over standard NEAT.

### Key files changed
- `src/tensorneat/genome/operations/mutation/ha_neat.py` — full implementation
- `src/tensorneat/genome/gene/conn/origin_conn.py` — `OriginConn` gene for historical markers

### Notes
Requires `OriginConn` and `activation_replace_rate=0.0` on `BiasNode`. Activation options: tanh, sigmoid, relu, sin, identity.

---

## [2026-03-08] Multi-Task Brax Evaluation

**Branch:** `dev/multi-task`
**Status:** Complete (merged to main)

### What
`MultiTaskBraxEnv` evaluates a single shared network across multiple Brax environments simultaneously. Handles mismatched I/O via zero-padded observations and sliced actions. Fitness is a weighted sum of per-task returns.

### Why
The thesis requires one network to solve Hopper + Walker2D at the same time. This is the evaluation harness for that setup.

### Key files changed
- `src/tensorneat/problem/rl/multi_task.py` — `MultiTaskBraxEnv`, `TaskSpec`, `FitnessAggregator`, `WeightedSum`
- `examples/brax/multi_task_hopper_walker.py` — reference example

### Notes
Network sized to largest task: 17 inputs (Walker2D), 6 outputs. Hopper gets zero-padded obs and first 3 actions.

---

## [2026-03-05] MLflow Tracking Integration

**Branch:** `dev/multi-task`
**Status:** Complete (merged to main)

### What
Integrated MLflow into the pipeline for per-generation metric logging. Added `per_task_tracking` flag to log per-task fitness separately. Added NEAT-specific metrics: species count, stagnation, genome complexity.

### Why
Need experiment tracking to compare NEAT vs HA-NEAT runs across seeds, population sizes, and generation counts.

### Key files changed
- `src/tensorneat/pipeline.py` — MLflow integration, `per_task_evaluate()` call, NEAT-specific metrics
