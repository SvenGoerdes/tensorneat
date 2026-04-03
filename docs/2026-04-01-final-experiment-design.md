# Final Experiment Design: NEAT vs HA-NEAT

**Date:** 2026-04-01
**Experiment name:** `multi_task_neat_vs_haneat_final`

## Scientific Question

Does HA-NEAT (heterogeneous activation functions) outperform standard NEAT (single tanh activation) on a multi-task Brax locomotion problem (Hopper + Walker2D)?

The ablation experiment (`multi_task_neat_vs_haneat_ablation`) running in parallel answers whether the performance difference is due to activation diversity specifically vs. the HA-NEAT machinery (OriginConn + HANEATMutation). This final run is the primary head-to-head comparison.

## Config

```yaml
experiment_name: "multi_task_neat_vs_haneat_final"
backend: mjx

tasks:
  - env_name: hopper
    max_step: 1000
    obs_size: 11
    act_size: 3
    weight: 1.0
  - env_name: walker2d
    max_step: 1000
    obs_size: 17
    act_size: 6
    weight: 1.0

algorithm_type: [neat, ha_neat]
aggregation_mode: normalized_min
pop_size: 4096
species_size: 20
survival_threshold: 0.3
compatibility_threshold: [0.3, 0.5]
compatibility_disjoint: 1.0
max_stagnation: 40
species_number_calculate_by: fitness
max_nodes: 50
max_conns: 200
activation_mutate_rate: 0.1
activation_options: [tanh, sigmoid, relu, sin, identity]
generation_limit: 2000
fitness_target: 10000
seed: [42, 123, 75, 7, 21]
mlflow_tracking: true
per_task_tracking: true
```

## Sweep Dimensions

| Dimension | Values | Count |
|---|---|---|
| algorithm_type | neat, ha_neat | 2 |
| compatibility_threshold | 0.3, 0.5 | 2 |
| seed | 42, 123, 75, 7, 21 | 5 |
| **Total runs** | | **20** |

## Rationale

- **pop_size 4096**: Larger population = more topological diversity, stronger selection pressure. Matches v4.
- **generation_limit 2000**: v4 runs showed fitness plateaus followed by late structural breakthroughs. 2000 gens gives room for these jumps. Early stopping at `fitness_target=10000` prevents wasted time on converged runs.
- **5 seeds**: Stronger statistical power for thesis claims. Seeds are arbitrary (42, 123, 75, 7, 21).
- **compatibility_threshold [0.3, 0.5]**: v3 used 0.5 (down from v2's 1.0). Testing 0.3 explores finer species partitioning. More species may correlate with better exploration.
- **compatibility_disjoint 1.0**: Fixed at best value from v2 sweep.
- **Dedicated GPU**: Run on second GPU to avoid contention with ablation experiment.

## Estimated Runtime

~15h per run × 20 runs = ~300h (~12.5 days) on a dedicated GPU.

## How to Run

Once the ablation experiment finishes and config.yaml is updated:

```bash
nohup uv run python -u main.py > output_final.log 2>&1 &
```

Monitor with:
```bash
tail -f output_final.log
grep "Run [0-9]" output_final.log
```
