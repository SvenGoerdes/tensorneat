# TensorNEAT + HA-NEAT

GPU-accelerated NEAT built on JAX, extended with **HA-NEAT** (Heterogeneous Activation NEAT) and **multi-task evaluation** for comparing NEAT vs HA-NEAT on Brax locomotion tasks.

## Project Goal

Compare standard NEAT against HA-NEAT on multi-task Brax environments (Hopper + Walker2D). A single evolved network must solve both tasks simultaneously using zero-padded observations and sliced actions.

## Build & Run

**Backend**: Training uses `mjx` (MuJoCo on GPU) by default. Set `backend:` in `config.yaml` to change. The `generalized` backend is faster but allows Walker2D floor-penetration exploits.

```bash
# Install (editable mode)
uv pip install -e .

# Run experiment grid (primary entrypoint)
uv run python main.py --config config.yaml

# Run tests
uv run pytest test/
```

Package source is in `src/tensorneat/`, configured via `pyproject.toml` with `setuptools`.
Python >= 3.10 required. Key deps: jax, brax, gymnax, flax, optax, networkx, sympy, mlflow.

## Architecture

Four-layer design: **Pipeline > Algorithm > Genome > Problem**

### Pipeline (`pipeline.py`)
Orchestrates evolution: `setup(state) -> step(state) -> auto_run(state)`.
Each step: `ask` (get population) -> `transform` -> `evaluate` -> `tell` (update).

- `mlflow_tracking=True`: Logs per-generation metrics (fitness stats, species count, genome complexity, stagnation)
- `per_task_tracking=True`: Logs per-task fitness for multi-task problems via `per_task_evaluate()`

### Algorithm (`algorithm/`)
- **NEAT** (`neat/neat.py`): Speciation-driven evolution with `SpeciesController`

Base class: `BaseAlgorithm` — implement `ask()`, `tell()`, `transform()`, `forward()`

### Genome (`genome/`)
- **DefaultGenome**: Feedforward networks with topological sort
- **Gene types** (`genome/gene/`): `DefaultNode`, `BiasNode`, `DefaultConn`, `OriginConn`
- **Operations** (`genome/operations/`): `DefaultMutation`, `HANEATMutation`, `DefaultCrossover`, `DefaultDistance`

Base class: `BaseGenome` — implement `transform()`, `forward()`, `initialize()`

### Problem (`problem/`)
- **RL** (`rl/`): `BraxEnv`, `MultiTaskBraxEnv`

Base class: `BaseProblem` — implement `evaluate()`, `input_shape`, `output_shape`

## HA-NEAT (`genome/operations/mutation/ha_neat.py`)

HA-NEAT extends NEAT with per-node activation function diversity and speciation protection. Implemented entirely in `HANEATMutation`, a subclass of `DefaultMutation`.

**Three key mechanisms:**
1. **Random activation on node creation** — new nodes get a random activation (tanh, sigmoid, relu, sin, identity) instead of identity
2. **One-node-per-generation activation mutation** — with probability `activation_mutate_rate` (default 0.1), one random hidden node's activation is changed. All connections touching that node get new historical markers, pushing the mutant into a protected species
3. **Prevention of standard activation mutation** — overrides `mutate_values()` to restore activation column after parent mutation, ensuring only the controlled mechanism modifies activations

**Required:** `OriginConn` gene (tracks historical markers) and `activation_replace_rate=0.0` on node gene.

## Multi-Task Evaluation (`problem/rl/multi_task.py`)

`MultiTaskBraxEnv` evaluates a single shared network across multiple RL environments.

**I/O handling:** Network sized to largest task (max obs/act dimensions). Smaller tasks get zero-padded observations and sliced actions.

**Fitness aggregation** is auto-selected at construction:
- If any `TaskSpec.max_reward != 1.0` → uses `NormalizedWeightedSum` (divides each task's fitness by its `max_reward` before weighting)
- Otherwise → uses plain `WeightedSum`
- Pass an explicit `aggregator=` to override

**`BRAX_REFERENCE_REWARDS`**: `hopper=3000`, `walker2d=5000`, `ant=6000`, `halfcheetah=8000`, `humanoid=6000`, `reacher=5`. `build_tasks()` in `main.py` looks these up automatically.

**Per-task tracking** logs two metrics per task per generation:
- `fitness/<env_name>` — raw episode return
- `fitness_normalized/<env_name>` — raw / max_reward (0–1 scale)

## Experiment Runner (`main.py`)

Primary entrypoint. Reads `config.yaml`, builds a parameter grid via Cartesian product of list-valued keys, and runs each combination sequentially.

Any key with a list value (except `tasks`, `activation_options`, `experiment_name`, `mlflow_tracking`, `per_task_tracking`) becomes a sweep dimension.

```yaml
algorithm_type: [neat, ha_neat]   # sweep
pop_size: [512, 1024]             # sweep
seed: [42, 123]                   # sweep
generation_limit: [100, 200]      # sweep
# → 16 total runs
```

Results saved to `results/<experiment_name>/<run_name>.npz` containing `nodes`, `conns`, `fitness`.

## Experimental Setup

**Comparison conditions:**
1. **Multi-Task NEAT** (baseline): Single tanh activation, standard mutation
2. **Multi-Task HA-NEAT** (treatment): 5 activation functions, historical marker reassignment, speciation protection

**Environments:** Hopper (11 obs, 3 act) + Walker2D (17 obs, 6 act)
**Network:** 17 inputs → 6 outputs (zero-padding/slicing for Hopper)
**Tracking:** MLflow per-generation metrics + per-task fitness trajectories

## Key Directories

```
main.py                # Primary entrypoint: config-driven experiment grid runner
config.yaml            # Experiment config (sweep over algorithm_type, pop_size, seed, etc.)
results/               # Output directory: best genomes saved as .npz per run
src/tensorneat/        # Library source
  pipeline.py          # Main orchestrator (+ MLflow tracking)
  algorithm/           # NEAT
  genome/              # Network representations + operations
    operations/mutation/ha_neat.py  # HA-NEAT mutation operator
  problem/             # Evaluation environments
    rl/multi_task.py   # Multi-task evaluation
  common/              # State, utilities
test/                  # pytest tests
```
