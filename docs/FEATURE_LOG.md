# Feature Log

Running record of features implemented for the NEAT vs HA-NEAT thesis comparison.

---

## [2026-03-26] Selection Pressure Tuning for Multi-Task NEAT

**Branch:** `main`
**Status:** Complete

### What
Adjusted three NEAT selection pressure parameters to reduce task-specialisation and give multi-task generalist networks a better chance of survival.

### Why
Analysis of MJX runs showed that `survival_threshold=0.1`, `max_stagnation=15`, and linear rank species allocation caused the evolution to eliminate balanced multi-task species too aggressively — species attempting to solve both tasks simultaneously were starved of offspring and killed off before they could develop the required structure.

### Key files changed
- `config.yaml` — `survival_threshold` 0.1→0.3, `max_stagnation` 15→40, `species_number_calculate_by` rank→fitness, `aggregation_mode` fixed to `normalized_min`, `generation_limit` 200→500, `pop_size` [512,1024,2048]→[1024,2048], `seed` [42,123]
- `main.py` — `max_stagnation` and `species_number_calculate_by` now read from config and passed to NEAT; both logged as MLflow params

### Notes
`species_number_calculate_by: fitness` uses proportional allocation based on absolute fitness differences rather than rank. With `normalized_min` aggregation, all species cluster in a narrow fitness band (0.3–0.4), so proportional allocation gives near-equal offspring counts — preventing any single species from dominating. Linear rank would still give the best species 3× more offspring than the worst even when fitness differences are tiny.

---

## [2026-03-19] MLflow Experiment Organisation & Fitness Aggregation Modes

**Branch:** `main`
**Status:** Complete

### What
Three improvements to MLflow tracking and fitness aggregation:
1. Runs now land in a named MLflow experiment (from `experiment_name` in config) instead of the Default experiment.
2. NEAT/HA-NEAT hyperparameters (`species_size`, `survival_threshold`, `compatibility_threshold`, `max_nodes`, `max_conns`, `activation_mutate_rate`, `activation_options`) are now logged as MLflow params per run.
3. Two new fitness aggregators added — `NormalizedMin` and `NormalizedProduct` — selectable via `aggregation_mode` in config.

### Why
Runs were piling up in MLflow's Default experiment making comparison difficult. Missing hyperparams made it impossible to filter runs in the UI. `NormalizedMin` forces evolution to improve the weakest task (prevents Hopper dominating Walker2D); `NormalizedProduct` collapses fitness when any task is near zero.

### Key files changed
- `src/tensorneat/pipeline.py` — `mlflow_experiment_name`, `mlflow_extra_params` params; compiled `per_task_evaluate` for performance; `mlflow.set_experiment()` call
- `src/tensorneat/problem/rl/multi_task.py` — `NormalizedMin`, `NormalizedProduct` aggregators; `aggregation_mode` param on `MultiTaskBraxEnv`
- `main.py` — passes `aggregation_mode` and extra MLflow params through
- `config.yaml` — added `aggregation_mode` key

### Notes
`aggregation_mode` can be swept: `[normalized_sum, normalized_min]` creates a grid dimension. Per-task evaluation is now JIT-compiled at startup, eliminating the per-generation Python overhead that caused >5s inter-generation delays.

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
