# TensorNEAT + HA-NEAT

GPU-accelerated NEAT built on JAX, extended with **HA-NEAT** (Heterogeneous Activation NEAT) and **multi-task evaluation** for comparing NEAT vs HA-NEAT on Brax locomotion tasks.

## Project Goal

Compare standard NEAT against HA-NEAT on multi-task Brax environments (Hopper + Walker2D). A single evolved network must solve both tasks simultaneously using zero-padded observations and sliced actions.

## Build & Run

```bash
uv pip install -e .
uv run python main.py --config config.yaml
uv run pytest test/
```

Python >= 3.10. Key deps: jax, brax, gymnax, flax, optax, networkx, sympy, mlflow.
Use `mjx` backend (default). `generalized` is faster but allows Walker2D floor-penetration exploits.

## Architecture

Four-layer design: **Pipeline > Algorithm > Genome > Problem**

- **Pipeline** (`pipeline.py`): `setup → step → auto_run`. MLflow tracking + per-task fitness breakdown.
- **Algorithm** (`algorithm/neat/neat.py`): Speciation-driven NEAT with `SpeciesController`.
- **Genome** (`genome/`): `DefaultGenome` with topological sort. Gene types: `DefaultNode`, `BiasNode`, `DefaultConn`, `OriginConn`. Mutations: `DefaultMutation`, `HANEATMutation`.
- **Problem** (`problem/rl/`): `BraxEnv`, `MultiTaskBraxEnv`. Fitness aggregation: `normalized_min` = `min(norm_hopper, norm_walker2d)`.

See `REFERENCE.md` for internals (HA-NEAT mechanism, pipeline details, aggregation logic).

## Experimental Setup

| Condition | Algorithm | Activations | Notes |
|---|---|---|---|
| Baseline | NEAT | tanh only | Standard mutation |
| Treatment | HA-NEAT | tanh, sigmoid, relu, sin, identity | Historical marker reassignment, speciation protection |
| Ablation | HA-NEAT (tanh only) | tanh only | Tests if activation diversity is the active ingredient |

**Environments:** Hopper (11 obs, 3 act) + Walker2D (17 obs, 6 act)
**Network:** 17 inputs → 6 outputs (zero-padding/slicing for Hopper)

## Experiment History

| Experiment | Key config | Outcome |
|---|---|---|
| `multi_task_neat_vs_haneat` | generalized backend, no normalization | Walker2D exploit via floor penetration |
| `multi_task_neat_vs_haneat_mjx_backend` | mjx, normalized_sum/min/product, pop 512-2048, 100-200 gen | Baseline MJX runs |
| `multi_task_neat_vs_haneat_mjx_backend_v2` | mjx, normalized_min, pop 1024+2048, 500 gen, disjoint sweep [0.5,1.0] | Best run: neat pop2048 disjoint1.0 seed123, fitness=0.43, 11 species |
| `multi_task_neat_vs_haneat_v3` | mjx, normalized_min, pop 2048, 500 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |
| `multi_task_neat_vs_haneat_v4` | mjx, normalized_min, pop 4096, 1000 gen, compat_threshold=0.5 | 4 runs (neat+ha_neat × seed42+123) |
| `multi_task_neat_vs_haneat_ablation` | mjx, normalized_min, pop 4096, 1000 gen, compat_threshold=0.5, ha_neat tanh only | 6 runs. No stat. sig. difference (permutation p=0.70, Cohen's d small) |
| `multi_task_neat_vs_haneat_final` | mjx, normalized_min, pop 4096, 2000 gen, compat_threshold=[0.3,0.5], 5 seeds | 10 NEAT + 4 HA-NEAT (server exp 1) |
| `multi_task_haneat_finalv2` | same config, ha_neat only, second GPU | 10 HA-NEAT (server exp 2). Combined with above for full comparison. |

**Final experiment genome counts (server):**
- NEAT: 10 seeds (5 × compat0.3 + 5 × compat0.5), all FINISHED
- HA-NEAT: 14 seeds total across both experiments (dedup needed for compat0.3 duplicates)

## Key Directories

```
main.py                      # Config-driven experiment grid runner
config.yaml                  # Current experiment config
eval/
  evaluate_genomes.py        # Batch re-evaluate .npz genomes → JSON
  outputs/                   # JSON re-evaluation results
results/                     # Best genomes saved as .npz per run
text/
  results_and_discussion.md  # Thesis Results & Discussion draft
  figures/                   # Plotting + stats scripts
    plot_training_curves.py       # Training curves from MLflow SQLite
    plot_evaluation_results.py    # Bar plots from re-evaluation JSON
    statistical_test.py           # Mann-Whitney U, permutation test, Cohen's d
src/tensorneat/              # Library source
docs/FEATURE_LOG.md          # Running log of all implemented features
REFERENCE.md                 # Detailed implementation notes
```

