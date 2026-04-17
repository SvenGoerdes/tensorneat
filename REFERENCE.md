# TensorNEAT Reference

Detailed implementation notes. See `CLAUDE.md` for the high-level overview.

## Pipeline internals (`pipeline.py`)

Orchestrates evolution: `setup(state) -> step(state) -> auto_run(state)`.
Each step: `ask` (get population) -> `transform` -> `evaluate` -> `tell` (update).

- `mlflow_tracking=True`: Logs per-generation metrics (fitness stats, species count, genome complexity, stagnation)
- `per_task_tracking=True`: Uses `evaluate_with_breakdown()` during the population vmap to capture per-task fitnesses **without re-evaluation**, avoiding XLA non-determinism. Logs `fitness/<env>`, `fitness_normalized/<env>`, `best_fitness/<env>`, `best_fitness_normalized/<env>` per generation.
- `step()` returns a 4-tuple `(state, pop, fitnesses, all_per_task)` when breakdown is used, 3-tuple otherwise. `auto_run()` handles both.

## HA-NEAT mechanism (`genome/operations/mutation/ha_neat.py`)

Implemented entirely in `HANEATMutation`, a subclass of `DefaultMutation`.

**Three key mechanisms:**
1. **Random activation on node creation** — new nodes get a random activation (tanh, sigmoid, relu, sin, identity) instead of identity
2. **One-node-per-generation activation mutation** — with probability `activation_mutate_rate` (default 0.1), one random hidden node's activation is changed. All connections touching that node get new historical markers, pushing the mutant into a protected species
3. **Prevention of standard activation mutation** — overrides `mutate_values()` to restore activation column after parent mutation, ensuring only the controlled mechanism modifies activations

**Required:** `OriginConn` gene (tracks historical markers) and `activation_replace_rate=0.0` on node gene.

## Multi-Task Evaluation internals (`problem/rl/multi_task.py`)

**I/O handling:** Network sized to largest task (max obs/act dimensions). Smaller tasks get zero-padded observations and sliced actions.

**Fitness aggregation** is auto-selected at construction:
- If any `TaskSpec.max_reward != 1.0` → uses `NormalizedWeightedSum`
- Otherwise → uses plain `WeightedSum`
- Pass an explicit `aggregator=` to override
- `aggregation_mode: normalized_min` → `NormalizedMin`: uses `min(norm_hopper, norm_walker2d)`, forces improvement on the weakest task

**`evaluate_with_breakdown(state, randkey, act_func, params)`** returns `(aggregate_scalar, per_task_array)` from a single evaluation — used by the pipeline to avoid XLA non-determinism from re-evaluation.

**`BRAX_REFERENCE_REWARDS`**: `hopper=3000`, `walker2d=5000`, `ant=6000`, `halfcheetah=8000`, `humanoid=6000`, `reacher=5`.

**Per-task tracking** logs four metrics per task per generation:
- `fitness/<env_name>` — raw episode return for best genome this generation
- `fitness_normalized/<env_name>` — raw / max_reward (0–1 scale)
- `best_fitness/<env_name>` — all-time best genome's raw return (monotonically non-decreasing)
- `best_fitness_normalized/<env_name>` — all-time best genome's normalized return

**Training fitness is stochastic (single episode per genome per generation).** Re-evaluation with different random seeds will differ, especially for Walker2D which has high episode variance. Always re-evaluate over multiple episodes (≥10) for true performance.

## Experiment Runner (`main.py`)

Reads `config.yaml`, builds a parameter grid via Cartesian product of list-valued keys, and runs each combination sequentially.

Any key with a list value (except `tasks`, `activation_options`, `experiment_name`, `mlflow_tracking`, `per_task_tracking`) becomes a sweep dimension. `backend` is also structural (not swept).

**Known bug in `run_single()`:** the run name uses `_gen{compatibility_disjoint}` instead of `_disjoint{compatibility_disjoint}`, producing filenames like `..._gen500_gen1.0_seed42.npz`. Cosmetic only — doesn't affect training.

**`compatibility_threshold`** is the primary lever for species count. Lower = finer partitioning = more species. EDA on v2 showed positive correlation between species count and fitness. Final experiment uses `[0.3, 0.5]`.

**`compatibility_disjoint`** controls how strongly structural differences push genomes into new species. Fixed at `1.0`.

## Ablation Experiment details

The ablation (`ha_neat_ablation`) uses full HA-NEAT machinery (OriginConn, HANEATMutation, marker reassignment) but restricts `activation_options` to `[tanh]` only. Tests whether activation diversity is the active ingredient, or whether the speciation/marker machinery alone provides benefit.

Key finding: even with tanh-only activations, OriginConn still reassigns historical markers on activation mutation (tanh→tanh is a no-op but markers change), pushing functionally identical genomes into separate species. Ablation had slightly *fewer* species than NEAT (avg 18.4 vs 19.2), opposite of expected.

Statistical results (n=3 vs n=3): permutation p=0.70, Cohen's d small. No significant difference.

## Re-evaluation & Analysis Pipeline

```bash
# 1. Re-evaluate saved genomes (30 episodes per genome)
python -u eval/evaluate_genomes.py --results_dir results/<experiment> --n_eval 30 --backend mjx

# 2. Plot training curves from MLflow
python text/figures/plot_training_curves.py --experiment_name <experiment> --db_path mlflow.db

# 3. Plot re-evaluation bar charts
python text/figures/plot_evaluation_results.py --input eval/outputs/<file>.json

# 4. Statistical tests (seed-level means)
python text/figures/statistical_test.py --input eval/outputs/<file>.json --algo_a neat --algo_b ha_neat
```

`evaluate_genomes.py` detects algorithm from filename: `ha_neat_ablation_*` → `ha_neat_ablation`, `ha_neat_*` → `ha_neat`, else `neat`. Order matters (most specific prefix first).

## Plot Style

NEAT = `#2176AE` (blue), HA-NEAT / ablation = `#D64933` (red). Serif font, size 11 labels, 9 ticks, no titles, remove top/right spines, DPI 300.
