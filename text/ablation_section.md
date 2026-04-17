# Ablation study: isolating the effect of activation diversity

HA-NEAT differs from standard NEAT in two ways, not one. The obvious difference is activation diversity: each node carries one of five activation functions, and the mutation operator can replace it each generation. The less obvious difference is structural. HA-NEAT uses `OriginConn` instead of the default connection gene, and whenever a node's activation changes, historical markers on all connected edges are reassigned, pushing the mutant into a new species. The question is whether `OriginConn` and this marker logic independently affect speciation dynamics (different species counts, different selection pressure, different population structure) without any activation diversity involved. If they do, a performance gap between NEAT and full HA-NEAT in the main experiment would be at least partly a structural artefact rather than a consequence of the activation functions.

To check this, we ran HA-NEAT with `activation_options=[tanh]` only. The mutation operator fires at rate 0.1, samples a random activation, and reassigns markers as normal, but with one option available, the sampled activation is always tanh, so nothing actually changes. `OriginConn` and marker reassignment run identically to the full HA-NEAT condition. Activation diversity is the only thing removed. We used three seeds (42, 123, 75), pop_size=1024, 500 generations, normalized_min aggregation, MJX backend. After training, each saved genome was evaluated over 10 independent episodes per task, and we report the mean.

| Condition | Seed | Hopper | Walker2D | Hopper (norm) | Walker2D (norm) |
|---|---|---|---|---|---|
| NEAT | 42 | 878.6 | 1994.1 | 0.293 | 0.399 |
| NEAT | 123 | 1120.5 | 816.6 | 0.373 | 0.163 |
| NEAT | 75 | 1026.3 | 1390.5 | 0.342 | 0.278 |
| **NEAT mean +/- std** | | **1008.5 +/- 99.6** | **1400.4 +/- 480.8** | **0.336 +/- 0.033** | **0.280 +/- 0.096** |
| HA-NEAT ablation | 42 | 1058.3 | 1356.8 | 0.353 | 0.271 |
| HA-NEAT ablation | 123 | 1168.2 | 1995.8 | 0.389 | 0.399 |
| HA-NEAT ablation | 75 | 1020.7 | 519.4 | 0.340 | 0.104 |
| **HA-NEAT ablation mean +/- std** | | **1082.4 +/- 62.6** | **1290.7 +/- 604.5** | **0.361 +/- 0.021** | **0.258 +/- 0.121** |

*Normalised by reference rewards: Hopper=3000, Walker2D=5000. Scores are means over 10 evaluation episodes per task.*

The two conditions produce similar results. On Hopper, the mean difference is 74 raw reward points, well inside one standard deviation for either condition. On Walker2D, the within-condition standard deviations (480 and 604 respectively) are several times larger than the gap between means (110), so seed-to-seed variation dominates entirely. Mann-Whitney U tests (two-sided, n=3 per group) returned p=0.700 on Hopper, p=1.000 on Walker2D, and p=1.000 on the aggregate normalized_min score (NEAT: 0.245 +/- 0.058, ablation: 0.255 +/- 0.117). With only three seeds per condition, these tests have little power to detect moderate effect sizes, so the p-values alone cannot establish equivalence. What they do reflect is that within-condition variance is considerably larger than between-condition variance, which is consistent with both algorithms drawing from the same performance distribution.

Three seeds is a small sample and this result should be read accordingly. Replication with more seeds would give a clearer picture. That said, the structural components of HA-NEAT (`OriginConn`, marker reassignment, `HANEATMutation`) do not appear to shift performance independently. The per-seed scores of NEAT and HA-NEAT ablation overlap substantially across both tasks. This supports treating any performance difference between NEAT and full HA-NEAT in the main experiment as a consequence of activation diversity rather than a side effect of the underlying implementation.
