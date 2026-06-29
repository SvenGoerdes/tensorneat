# pop200 — Three-Way Algorithm Comparison (NEAT vs HA-NEAT vs Ablation)

_Generated 2026-06-15 22:21. Source: `analysis/outputs/pop200/combined.json`._

## Setup

- **Experiment:** `multi_task_neat_vs_haneat_pop200` (pop = 200, gen = 750, species_size = 10, compat = 0.3, aggregation = `normalized_min`, MJX backend).
- **Conditions (seeds re-evaluated):** NEAT = 30 / HA-NEAT = 30 / Ablation = 30.
- **Re-evaluation:** 20 independent episodes per genome per task on hopper + walker2d (MJX), seeds distinct from training rollouts.
- **Reference rewards (normalization):** hopper = 3000, walker2d = 5000.
- **Headline metric:** `normalized_min` = min(reward_hopper / 3000, reward_walker2d / 5000) — the multi-task objective optimized during training.

Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant.

## Headline metric: normalized_min

### normalized_min (per-seed)

| Condition | n | Mean | SD | SEM | 95% CI | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| NEAT | 30 | 0.188 | 0.088 | 0.016 | [0.155, 0.221] | 0.183 | 0.129 | 0.027 | 0.407 |
| HA-NEAT | 30 | 0.214 | 0.084 | 0.015 | [0.183, 0.245] | 0.208 | 0.124 | 0.061 | 0.369 |
| HA-NEAT (ablation, tanh-only) | 30 | 0.209 | 0.080 | 0.015 | [0.179, 0.239] | 0.199 | 0.107 | 0.088 | 0.367 |

**Kruskal-Wallis omnibus:** H = 1.483, p = 0.4764 ns, ε² = -0.006

Pairwise contrasts (Holm-Bonferroni corrected across the 3 contrasts):

| Contrast | Mean diff | Cohen's d (mag) | MWU U | p (MWU) | p Holm | Perm p | Perm Holm |
|---|---|---|---|---|---|---|---|
| NEAT vs HA-NEAT | -0.026 | -0.301 (small) | 368.0 | 0.2282 ns | 0.6847 ns | 0.2478 | 0.7434 |
| NEAT vs Ablation | -0.022 | -0.256 (small) | 394.0 | 0.4119 ns | 0.8238 ns | 0.3250 | 0.7434 |
| HA-NEAT vs Ablation | 0.004 | 0.053 (negligible) | 467.0 | 0.8073 ns | 0.8238 ns | 0.8409 | 0.8409 |

## Secondary: per-task performance

### Hopper (raw reward)

Normalized by 3000.

| Condition | n | Mean | SD | SEM | 95% CI | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| NEAT | 30 | 1025.233 | 53.027 | 9.681 | [1005.433, 1045.034] | 1024.771 | 16.001 | 845.670 | 1229.877 |
| HA-NEAT | 30 | 971.862 | 160.183 | 29.245 | [912.048, 1031.675] | 1014.223 | 31.875 | 272.328 | 1105.743 |
| HA-NEAT (ablation, tanh-only) | 30 | 1015.631 | 71.961 | 13.138 | [988.760, 1042.502] | 1024.976 | 28.751 | 825.612 | 1134.908 |

**Kruskal-Wallis omnibus:** H = 4.786, p = 0.0913 ns, ε² = 0.032

Pairwise contrasts (Holm-Bonferroni corrected across the 3 contrasts):

| Contrast | Mean diff | Cohen's d (mag) | MWU U | p (MWU) | p Holm | Perm p | Perm Holm |
|---|---|---|---|---|---|---|---|
| NEAT vs HA-NEAT | 53.372 | 0.447 (small) | 593.0 | 0.0351 * | 0.1054 ns | 0.0722 | 0.2166 |
| NEAT vs Ablation | 9.602 | 0.152 (negligible) | 465.0 | 0.8303 ns | 0.8303 ns | 0.5651 | 0.5651 |
| HA-NEAT vs Ablation | -43.769 | -0.352 (small) | 340.0 | 0.1055 ns | 0.2109 ns | 0.1913 | 0.3826 |

### Walker2D (raw reward)

Normalized by 5000.

| Condition | n | Mean | SD | SEM | 95% CI | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| NEAT | 30 | 956.796 | 469.739 | 85.762 | [781.393, 1132.200] | 917.441 | 643.717 | 133.232 | 2037.004 |
| HA-NEAT | 30 | 1124.424 | 433.355 | 79.119 | [962.606, 1286.241] | 1135.259 | 702.811 | 305.928 | 1956.707 |
| HA-NEAT (ablation, tanh-only) | 30 | 1072.073 | 441.826 | 80.666 | [907.093, 1237.054] | 996.684 | 534.291 | 440.247 | 2075.643 |

**Kruskal-Wallis omnibus:** H = 2.146, p = 0.3419 ns, ε² = 0.002

Pairwise contrasts (Holm-Bonferroni corrected across the 3 contrasts):

| Contrast | Mean diff | Cohen's d (mag) | MWU U | p (MWU) | p Holm | Perm p | Perm Holm |
|---|---|---|---|---|---|---|---|
| NEAT vs HA-NEAT | -167.627 | -0.371 (small) | 348.0 | 0.1335 ns | 0.4004 ns | 0.1509 | 0.4527 |
| NEAT vs Ablation | -115.277 | -0.253 (small) | 396.0 | 0.4290 ns | 0.8579 ns | 0.3251 | 0.6502 |
| HA-NEAT vs Ablation | 52.350 | 0.120 (negligible) | 487.0 | 0.5895 ns | 0.8579 ns | 0.6504 | 0.6504 |

## Sanity check: training fitness vs re-evaluation

Training fitness is the single `normalized_min` value stored in each `.npz` at the end of training (one rollout). The re-evaluated `normalized_min` above averages 20 fresh episodes. Close agreement indicates the saved champions generalize across seeds.

### Training fitness (stored normalized_min)

| Condition | n | Mean | SD | SEM | 95% CI | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| NEAT | 30 | 0.335 | 0.033 | 0.006 | [0.323, 0.347] | 0.339 | 0.019 | 0.253 | 0.420 |
| HA-NEAT | 30 | 0.330 | 0.031 | 0.006 | [0.319, 0.342] | 0.340 | 0.016 | 0.256 | 0.383 |
| HA-NEAT (ablation, tanh-only) | 30 | 0.338 | 0.027 | 0.005 | [0.328, 0.347] | 0.342 | 0.009 | 0.272 | 0.390 |

**Kruskal-Wallis omnibus:** H = 1.377, p = 0.5025 ns, ε² = -0.007

Pairwise contrasts (Holm-Bonferroni corrected across the 3 contrasts):

| Contrast | Mean diff | Cohen's d (mag) | MWU U | p (MWU) | p Holm | Perm p | Perm Holm |
|---|---|---|---|---|---|---|---|
| NEAT vs HA-NEAT | 0.005 | 0.152 (negligible) | 428.0 | 0.7506 ns | 0.8238 ns | 0.5618 | 1.0000 |
| NEAT vs Ablation | -0.003 | -0.086 (negligible) | 374.0 | 0.2643 ns | 0.7930 ns | 0.7389 | 1.0000 |
| HA-NEAT vs Ablation | -0.007 | -0.258 (small) | 394.0 | 0.4119 ns | 0.8238 ns | 0.3092 | 0.9276 |

## Interpretation

- On the headline metric (`normalized_min`), the Kruskal-Wallis omnibus test finds **no significant difference** among the three conditions (H = 1.48, p = 0.476). The three algorithms perform equivalently on the multi-task objective they were trained on.
- Mean `normalized_min` by condition: NEAT = 0.188, HA-NEAT = 0.214, Ablation = 0.209 (highest: HA-NEAT).
- After Holm-Bonferroni correction, **none** of the three pairwise contrasts on `normalized_min` reach significance. Effect sizes are summarized in the table above.
- **Ablation read:** HA-NEAT vs HA-NEAT(tanh-only) differ by d = 0.05 (negligible), Holm p = 0.824. This isolates whether activation-function *diversity* (not the HA-NEAT machinery itself) is the active ingredient.
