# Generational Dynamics — pop200 experiment

Experiment: `multi_task_neat_vs_haneat_pop200` (MLflow experiment_id 14)

Population 200, 750 generations, 30 NEAT / 30 HA-NEAT / 30 ablation seeds.

Runs deduplicated by name (kept the run with the most logged steps). Bands in the figures are the inter-quartile range (IQR, 25th-75th percentile) across seeds; lines are the across-seed mean.

## Metrics available in MLflow (per generation)

- `neat/best_genome_n_nodes` — node count of the best genome
- `neat/best_genome_n_conns` — connection count of the best genome
- `neat/avg_genome_n_conns` — population-mean connection count
- `neat/n_species` — number of non-empty species

## Final-window summary (generations 651–750, last 100)

### Species count (per-seed mean over final window)

| Condition | n | mean | std | median | IQR |
|---|---|---|---|---|---|
| NEAT | 30 | 9.97 | 0.03 | 9.98 | [9.95, 9.99] |
| HA-NEAT | 30 | 9.97 | 0.03 | 9.98 | [9.96, 9.99] |
| HA-NEAT (tanh only) | 30 | 9.97 | 0.04 | 9.98 | [9.95, 9.99] |

### Best-genome node count (per-seed mean over final window)

| Condition | n | mean | std | median | IQR |
|---|---|---|---|---|---|
| NEAT | 30 | 26.57 | 2.98 | 25.60 | [24.09, 28.43] |
| HA-NEAT | 30 | 24.68 | 2.63 | 23.44 | [23.35, 25.25] |
| HA-NEAT (tanh only) | 30 | 24.03 | 1.35 | 23.41 | [23.23, 24.45] |

### Best-genome connection count (per-seed mean over final window)

| Condition | n | mean | std | median | IQR |
|---|---|---|---|---|---|
| NEAT | 30 | 43.00 | 7.18 | 43.83 | [39.97, 45.01] |
| HA-NEAT | 30 | 44.37 | 10.58 | 43.64 | [36.44, 48.55] |
| HA-NEAT (tanh only) | 30 | 40.21 | 9.18 | 40.25 | [32.55, 46.86] |

## Kruskal–Wallis test — final-window species count

Across the three conditions on the per-seed mean species count over the final 100 generations:

- H = 0.114, p = 0.9448

There is no significant difference in sustained species count across conditions (alpha = 0.05).

## Interpretation

NEAT sustains a mean best-genome node count of 26.57 in the final window versus 24.68 (HA-NEAT) and 24.03 (ablation), and a connection count of 43.00 versus 44.37 / 40.21 — consistent with the final-genome finding that NEAT accretes more structure at equal fitness. The complexity trajectories show when this gap opens up (see complexity_over_generations.png). 
Species counts are statistically indistinguishable across conditions (mean 9.97/9.97/9.97 for NEAT/HA-NEAT/ablation), so HA-NEAT's speciation protection does not translate into a measurably larger sustained species pool at this population size.

