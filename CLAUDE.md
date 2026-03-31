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
- `per_task_tracking=True`: Uses `evaluate_with_breakdown()` during the population vmap to capture per-task fitnesses **without re-evaluation**, avoiding XLA non-determinism. Logs `fitness/<env>`, `fitness_normalized/<env>`, `best_fitness/<env>`, `best_fitness_normalized/<env>` per generation.
- `step()` returns a 4-tuple `(state, pop, fitnesses, all_per_task)` when breakdown is used, 3-tuple otherwise. `auto_run()` handles both.

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
- `aggregation_mode: normalized_min` → `NormalizedMin`: uses `min(norm_hopper, norm_walker2d)`, forces improvement on the weakest task

**`evaluate_with_breakdown(state, randkey, act_func, params)`** returns `(aggregate_scalar, per_task_array)` from a single evaluation — used by the pipeline to avoid XLA non-determinism from re-evaluation.

**`BRAX_REFERENCE_REWARDS`**: `hopper=3000`, `walker2d=5000`, `ant=6000`, `halfcheetah=8000`, `humanoid=6000`, `reacher=5`. `build_tasks()` in `main.py` looks these up automatically.

**Per-task tracking** logs four metrics per task per generation:
- `fitness/<env_name>` — raw episode return for best genome this generation
- `fitness_normalized/<env_name>` — raw / max_reward (0–1 scale)
- `best_fitness/<env_name>` — all-time best genome's raw return (monotonically non-decreasing)
- `best_fitness_normalized/<env_name>` — all-time best genome's normalized return

**Training fitness is stochastic (single episode per genome per generation).** Re-evaluation with different random seeds will differ, especially for Walker2D which has high episode variance. The saved `.npz` genome's MLflow `best_fitness` is an optimistic single-episode estimate. Always re-evaluate over multiple episodes (≥10) for true performance.

## Experiment Runner (`main.py`)

Primary entrypoint. Reads `config.yaml`, builds a parameter grid via Cartesian product of list-valued keys, and runs each combination sequentially.

Any key with a list value (except `tasks`, `activation_options`, `experiment_name`, `mlflow_tracking`, `per_task_tracking`) becomes a sweep dimension. `backend` is also structural (not swept).

**Known bug in `run_single()`:** the run name uses `_gen{compatibility_disjoint}` instead of `_disjoint{compatibility_disjoint}`, producing filenames like `..._gen500_gen1.0_seed42.npz`. Cosmetic only — doesn't affect training.

**`compatibility_threshold`** (in `config.yaml`) is the primary lever for species count. Lower = finer partitioning = more species. EDA on v2 showed positive correlation between species count and fitness. v3 uses `0.5` (down from `1.0`).

**`compatibility_disjoint`** controls how strongly structural differences (new nodes/conns) push genomes into new species. Fixed at `1.0` in v3 (best in v2).

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
run_after.sh           # Launch main.py after N hours (usage: nohup bash run_after.sh 15 > run_after.log 2>&1 &)
evaluate_genomes.py    # Batch re-evaluate all saved .npz genomes (--results_dir, --backend, --n_eval)
evaluate_mjx.py        # Evaluate + visualize a single .npz genome (--backend, --tasks, --format)
results/               # Output directory: best genomes saved as .npz per run
notebooks/             # EDA notebooks
  eda_v2_hyperparams.ipynb  # Hyperparameter analysis for v2 runs (species, fitness, complexity)
src/tensorneat/        # Library source
  pipeline.py          # Main orchestrator (+ MLflow tracking)
  algorithm/           # NEAT
  genome/              # Network representations + operations
    operations/mutation/ha_neat.py  # HA-NEAT mutation operator
  problem/             # Evaluation environments
    rl/multi_task.py   # Multi-task evaluation + evaluate_with_breakdown()
  common/              # State, utilities
test/                  # pytest tests
docs/FEATURE_LOG.md    # Running log of all implemented features
```

## Experiment History

| Experiment | Key config | Outcome |
|---|---|---|
| `multi_task_neat_vs_haneat` | generalized backend, no normalization | Walker2D exploit via floor penetration |
| `multi_task_neat_vs_haneat_mjx_backend` | mjx, normalized_sum/min/product, pop 512-2048, 100-200 gen | Baseline MJX runs |
| `multi_task_neat_vs_haneat_mjx_backend_v2` | mjx, normalized_min, pop 1024+2048, 500 gen, disjoint sweep [0.5,1.0] | Best run: neat pop2048 disjoint1.0 seed123, fitness=0.43, 11 species |
| `multi_task_neat_vs_haneat_v3` | mjx, normalized_min, pop 2048, 500 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |
| `multi_task_neat_vs_haneat_v4` | mjx, normalized_min, pop 4096, 1000 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |

