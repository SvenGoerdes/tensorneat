# pop200 — Network Complexity of Final Champion Genomes

_Generated 2026-07-02 15:11 by `analysis/network_complexity_pop200.py`._

## Setup

- 90 champion genomes (`results/multi_task_neat_vs_haneat_pop200/*.npz`), 30 per condition.
- **Counting rules (verified on raw arrays):** genome arrays are fixed-size (50 nodes × 4, 200 conns) with NaN-padded unused rows; only non-NaN rows are counted. The node array *includes* the 17 input and 6 output nodes (indices 0–22); `n_hidden_nodes` = valid nodes − 23. Connections have no `enabled` flag in TensorNEAT (disabled connections are deleted), so every valid connection is active.
- Fitness = `normalized_min` from the 20-episode re-evaluation (`analysis/outputs/pop200/combined.json`).
- Effect size: Cliff's δ (thresholds 0.147/0.33/0.474) and rank-biserial correlation.

## Complexity by condition

### Hidden nodes

| Condition | n | Median [IQR] | Mean ± SD | Min–Max |
|---|---|---|---|---|
| NEAT | 30 | 2.0 [1.0, 3.0] | 2.2 ± 2.2 | 0–11 |
| HA-NEAT | 30 | 1.0 [0.0, 2.0] | 1.8 ± 3.2 | 0–15 |
| Ablation | 30 | 0.0 [0.0, 2.0] | 1.2 ± 1.9 | 0–8 |

**Kruskal-Wallis:** H = 7.451, p = 0.0241 *, ε² = 0.063

| Contrast | U | p (MWU) | p (Holm) | Cliff's δ (mag) | Rank-biserial |
|---|---|---|---|---|---|
| NEAT vs HA-NEAT | 579.5 | 0.0501 ns | 0.1003 ns | 0.288 (small) | -0.288 |
| NEAT vs Ablation | 619.5 | 0.0099 ** | 0.0297 * | 0.377 (medium) | -0.377 |
| HA-NEAT vs Ablation | 491.5 | 0.5171 ns | 0.5171 ns | 0.092 (negligible) | -0.092 |

### Connections

| Condition | n | Median [IQR] | Mean ± SD | Min–Max |
|---|---|---|---|---|
| NEAT | 30 | 48.5 [40.2, 60.2] | 52.3 ± 15.9 | 27–90 |
| HA-NEAT | 30 | 51.5 [44.2, 62.0] | 53.9 ± 15.1 | 30–93 |
| Ablation | 30 | 47.0 [38.8, 51.8] | 46.9 ± 11.8 | 24–71 |

**Kruskal-Wallis:** H = 2.819, p = 0.2443 ns, ε² = 0.009

| Contrast | U | p (MWU) | p (Holm) | Cliff's δ (mag) | Rank-biserial |
|---|---|---|---|---|---|
| NEAT vs HA-NEAT | 409.0 | 0.5491 ns | 0.6433 ns | -0.091 (negligible) | 0.091 |
| NEAT vs Ablation | 517.5 | 0.3217 ns | 0.6433 ns | 0.150 (small) | -0.150 |
| HA-NEAT vs Ablation | 564.0 | 0.0931 ns | 0.2792 ns | 0.253 (small) | -0.253 |

### Total nodes (incl. 17 inputs + 6 outputs)

| Condition | n | Median [IQR] | Mean ± SD | Min–Max |
|---|---|---|---|---|
| NEAT | 30 | 25.0 [24.0, 26.0] | 25.2 ± 2.2 | 23–34 |
| HA-NEAT | 30 | 24.0 [23.0, 25.0] | 24.8 ± 3.2 | 23–38 |
| Ablation | 30 | 23.0 [23.0, 25.0] | 24.2 ± 1.9 | 23–31 |

**Kruskal-Wallis:** H = 7.451, p = 0.0241 *, ε² = 0.063

| Contrast | U | p (MWU) | p (Holm) | Cliff's δ (mag) | Rank-biserial |
|---|---|---|---|---|---|
| NEAT vs HA-NEAT | 579.5 | 0.0501 ns | 0.1003 ns | 0.288 (small) | -0.288 |
| NEAT vs Ablation | 619.5 | 0.0099 ** | 0.0297 * | 0.377 (medium) | -0.377 |
| HA-NEAT vs Ablation | 491.5 | 0.5171 ns | 0.5171 ns | 0.092 (negligible) | -0.092 |

## Complexity vs fitness

### Spearman: fitness (normalized_min) vs n_connections

| Group | n | ρ | p |
|---|---|---|---|
| NEAT | 30 | -0.368 | 0.0453 * |
| HA-NEAT | 30 | -0.143 | 0.4496 ns |
| Ablation | 30 | 0.042 | 0.8260 ns |
| Pooled | 90 | -0.185 | 0.0815 ns |

### Spearman: fitness (normalized_min) vs n_hidden_nodes

| Group | n | ρ | p |
|---|---|---|---|
| NEAT | 30 | 0.326 | 0.0783 ns |
| HA-NEAT | 30 | -0.129 | 0.4984 ns |
| Ablation | 30 | 0.127 | 0.5051 ns |
| Pooled | 90 | 0.061 | 0.5648 ns |

## HA-NEAT hidden-node activation functions

Across the 30 HA-NEAT seeds there are **55 hidden nodes in total** (17/30 seeds evolved at least one hidden node).

| Activation | Hidden nodes | Share |
|---|---|---|
| tanh | 6 | 10.9% |
| sigmoid | 13 | 23.6% |
| relu | 15 | 27.3% |
| sin | 9 | 16.4% |
| identity | 12 | 21.8% |

## Figures

- `complexity_hidden_nodes.png`
- `complexity_connections.png`
- `complexity_fitness_vs_connections.png`
- `haneat_activation_distribution.png`

## Interpretation

- Hidden nodes: Kruskal-Wallis finds a significant difference across conditions (H = 7.45, p = 0.024); connections likewise no significant difference (H = 2.82, p = 0.244).
- Median hidden nodes: NEAT = 2.0, HA-NEAT = 1.0, Ablation = 0.0; median connections: NEAT = 48.5, HA-NEAT = 51.5, Ablation = 47.0.
- Pooled Spearman correlation between connections and fitness: ρ = -0.18 (p = 0.082).
- Given that fitness itself showed no significant difference across conditions (Kruskal-Wallis p = 0.476, see `report.md`), the structural comparison addresses whether HA-NEAT reaches equivalent fitness with *different* complexity — e.g., fewer hidden nodes because activation diversity substitutes for structure. The tables above show whether that substitution effect is present at this population scale.

## Suggested further analyses

- **Complexity over generations:** extract per-generation node/connection counts of the best genome from the MLflow metrics (or re-log them) to see whether HA-NEAT's structural growth *rate* differs even if the endpoints do not — bloat dynamics are invisible in final-champion snapshots.
- **Species counts over time:** pull `species_count` per generation from `mlflow.db` to test whether activation diversity sustains more species (diversity maintenance) under the shared compat threshold of 0.3.
- **Per-task fitness trade-off:** scatter norm_hopper vs norm_walker2d per champion and compare the trade-off frontier across conditions — conditions could reach equal normalized_min via different task balances.
- **Activation-function enrichment vs fitness within HA-NEAT:** correlate the fraction of non-tanh hidden nodes (or activation entropy) with fitness across the 30 HA-NEAT seeds to test whether activation diversity is actually exploited by better solutions.
