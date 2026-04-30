# Chapter 5: Results

<!-- ================================================================
     DRAFT — Skeleton with preliminary data from v3/v4/ablation runs.
     All [PLACEHOLDER: ...] markers must be replaced once the final
     experiment (multi_task_neat_vs_haneat_final, 20 runs) completes.
     All figure references are placeholders for figures to be generated
     from MLflow data; see docs/2026-04-01-final-experiment-design.md.
     ================================================================ -->

## 5.1 Experimental overview

Both NEAT and HA-NEAT get the same task: one network, two bodies. A single evolved genome must control a Hopper (11 observations, 3 actions) and a Walker2D (17 observations, 6 actions) using the same weights. The network is sized to Walker2D; Hopper observations are zero-padded to 17 dimensions, and the 6-dimensional output is sliced down to 3 for Hopper's actuators. Fitness is the normalized minimum of the two per-task scores. That matters: a genome that gets good at Hopper while ignoring Walker2D will score on Walker2D, the task it neglected. The aggregation selection study (Methodology, Section 3) landed on this after normalized-sum and normalized-product both let genomes specialize their way to a decent aggregate score. The rest of the chapter works through the results in layers — the primary experiment head-to-head, per-task breakdowns, a re-evaluation of whether training-time fitness holds up across multiple episodes, species and complexity dynamics, and the ablation study on activation diversity.

Three experiments make up the comparison. First, the main head-to-head: NEAT vs HA-NEAT across population sizes, generation counts, and random seeds. Second, a multi-episode re-evaluation of saved genomes, because training fitness comes from a single episode and may not reflect actual policy quality. Third, an ablation that runs HA-NEAT's full machinery with a single activation function, testing whether any advantage comes from activation diversity or from the structural changes around it. Section 5.2 reports aggregate performance, Section 5.3 breaks it down per task, Section 5.4 compares training estimates to re-evaluation, Section 5.5 covers species dynamics and genome complexity, and Section 5.6 presents the ablation.

All runs used the MJX (MuJoCo on GPU) physics backend, population sizes of 1024-4096, and between 500 and 2000 generations. [PLACEHOLDER: final hardware details, GPU type, JIT compile time, wall-clock per generation]. Random seeds were drawn from {42, 123, 75, 7, 21}; preliminary results use subsets of 2-3 seeds, and the final experiment uses all 5.

---

## 5.2 Multi-task performance comparison

We compared the best-found normalized-minimum fitness per run across both methods and several configurations. Table 1 collects results across experiment versions; Figure 1 shows the fitness training curves for the final experiment.

**Table 1. Aggregate normalized-minimum fitness per method across experiment versions.**

| Experiment | Method | Pop | Gens | Seeds | Scores (per seed) | Mean |
|---|---|---|---|---|---|---|
| v3 | NEAT | 2048 | 500 | 42, 123 | 0.358, 0.341 | 0.349 |
| v3 | HA-NEAT | 2048 | 500 | 42, 123 | 0.342, 0.484 | 0.413 |
| v4 | NEAT | 4096 | 1000 | 42 | 0.399 | 0.399 |
| v4 | HA-NEAT | 4096 | 1000 | 42, 123 | 0.473, 0.457 | 0.465 |
| final | NEAT | 4096 | 2000 | 42, 123, 75, 7, 21 | [PLACEHOLDER] | [PLACEHOLDER] |
| final | HA-NEAT | 4096 | 2000 | 42, 123, 75, 7, 21 | [PLACEHOLDER] | [PLACEHOLDER] |

<!-- Figure 1: Fitness training curves — aggregate fitness/max over generations for NEAT vs HA-NEAT.
     Source: MLflow experiment multi_task_neat_vs_haneat_final, metric fitness/max.
     Show individual seed traces (low opacity) + smoothed mean (bold).
     Two lines: NEAT (blue) vs HA-NEAT (orange).
     X-axis: generation, Y-axis: normalized_min fitness [0, 1]. -->

In the preliminary v3 runs (pop=2048, 500 generations), HA-NEAT averaged 0.413 across two seeds, compared to 0.349 for NEAT, a gap of 0.064 normalized units. Most of that gap came from one strong HA-NEAT run (seed 123: 0.484). In the v4 configuration (pop=4096, 1000 generations), the single completed NEAT run reached 0.399, while the two HA-NEAT runs averaged 0.465. The final experiment, with 5 seeds per method, will show whether this advantage holds with a larger sample. [PLACEHOLDER: final experiment mean +/- std for NEAT and HA-NEAT. Update this paragraph once complete.]

Across both experiment versions, HA-NEAT tended toward higher aggregate fitness, though with high cross-seed variance in both methods. The aggregate number alone does not tell us whether both tasks improved or whether one task carried the other.

---

## 5.3 Per-task analysis: Hopper and Walker2D

A moderate aggregate score can hide opposite extremes. A network that does well on Hopper but fails Walker2D still reports a nonzero normalized minimum. We split the aggregate into per-task fitness to see whether the two methods differ in which task they learn and which they neglect.

**Table 2. Per-task fitness for best genomes in v4 experiment.**
Training = single-episode estimate from MLflow. Re-eval = mean +/- std over 5 episodes (see Section 5.4). Normalized values in parentheses.

| Method | Seed | Hopper (train) | Hopper (re-eval) | Walker2D (train) | Walker2D (re-eval) |
|---|---|---|---|---|---|
| HA-NEAT | 42 | 1905 (0.635) | 1860 +/- 322 (0.620) | 2365 (0.473) | 2005 +/- 162 (0.401) |
| HA-NEAT | 123 | 1370 (0.457) | 752 +/- 435 (0.251) | 2330 (0.466) | 1953 +/- 483 (0.391) |
| NEAT | 42 | 1200 (0.400) | 1196 +/- 3 (0.399) | 1997 (0.399) | 1987 +/- 11 (0.397) |

<!-- Figure 2: Per-task training curves.
     Two panels side by side: left = Hopper fitness/task over generations, right = Walker2D.
     Source: MLflow metrics fitness/hopper and fitness/walker2d.
     Lines: NEAT (blue) vs HA-NEAT (orange), one trace per seed.
     Raw Y-axis (not normalized). Horizontal reference lines: Hopper=3000, Walker2D=5000. -->

NEAT seed42 produced almost perfectly balanced per-task scores: 1200 on Hopper (0.400 normalized) and 1997 on Walker2D (0.399 normalized). HA-NEAT seed42 was different. It scored 1905 on Hopper (0.635 normalized) but only 2365 on Walker2D (0.473 normalized). Walker2D was the bottleneck in the normalized-minimum aggregate, even though the raw Walker2D reward was higher than what NEAT achieved. [PLACEHOLDER: final experiment per-task breakdown. Update Table 2 and this paragraph.]

The training curves show [PLACEHOLDER: describe from Figure 2 once generated, e.g. whether Walker2D convergence lags Hopper, whether HA-NEAT's Hopper advantage appears early or late in training]. These numbers are single-episode training estimates though, and Section 5.4 shows they can be misleading.

---

## 5.4 Training fitness vs re-evaluation

Each genome is evaluated on a single episode per generation during training. That is a noisy estimate, especially for Walker2D, where episode outcomes depend on the initial physics state and contact sequence. To measure how noisy, we re-evaluated the best saved genomes from v4 over 5 episodes each with different random seeds.

<!-- Figure 3: Training vs re-evaluation bar chart.
     Two panels: left = Hopper, right = Walker2D.
     For each panel: grouped bars per run (HA-NEAT seed42, HA-NEAT seed123, NEAT seed42).
     Training-best bar (steel blue) vs re-eval mean +/- std bar (dark orange, with error cap).
     Y-axis: normalized fitness [0, 1]. Reference line at 1.0. -->

The gaps were large for HA-NEAT seed123 (Table 2, Figure 3). Its training-best Hopper fitness of 1370 dropped to 752 +/- 435 on re-evaluation, a 45% reduction with high episode variance. Walker2D held up better: 2330 during training, 1953 +/- 483 on re-evaluation. NEAT seed42 told a different story: re-eval scores were nearly identical to training (Hopper: 1200 vs 1196 +/- 3; Walker2D: 1997 vs 1987 +/- 11). This genome had found a stable, low-variance gait. HA-NEAT seed42 fell in between, with a moderate Hopper drop (1905 to 1860 +/- 322) and a larger Walker2D drop (2365 to 2005 +/- 162).

The training-best fitness stored in MLflow is an optimistic single-episode snapshot. For HA-NEAT genomes, multi-episode re-evaluation can change the picture substantially. Section 6.4 discusses the implications.

---

## 5.5 Species dynamics and genome complexity

HA-NEAT changes speciation dynamics as a side effect of its design. As described in Section 5.6, when a node's activation is mutated, all connections touching that node receive new historical markers, which increases the genetic distance between the mutant and its parent population. This could produce systematically more species than standard NEAT under otherwise identical settings. We compare species count and best-genome structural complexity over generations for both methods.

<!-- Figure 4: Species dynamics over generations.
     Source: MLflow metric neat/n_species.
     Two lines: NEAT (blue) vs HA-NEAT (orange), per seed + smoothed mean.
     X-axis: generation, Y-axis: species count.
     Use final experiment data; placeholder: v4 data. -->

<!-- Figure 5: Genome complexity over generations.
     Two panels: left = best_genome_n_nodes, right = best_genome_n_conns.
     Source: MLflow metrics neat/best_genome_n_nodes, neat/best_genome_n_conns.
     Same line style as Figure 4. -->

[PLACEHOLDER: Describe Figure 4 findings. Does HA-NEAT produce more species than NEAT? Prior EDA on v2 data found a positive correlation between species count and fitness. If HA-NEAT's marker reassignment inflates species count, this could be a confound when attributing performance gains to activation diversity.]

[PLACEHOLDER: Describe Figure 5 findings. Does HA-NEAT converge to simpler or more complex network topologies than NEAT? Activation diversity might reduce the pressure to add hidden nodes, since one activation change can transform a node's computational role without topological change.]

Whether these species and complexity patterns reflect a real advantage or a confound is the question the ablation (Section 5.6) was designed to answer.

---

## 5.6 Ablation: activation diversity vs HA-NEAT machinery

HA-NEAT changes two things relative to standard NEAT. First, it draws per-node activations from a set of five functions (tanh, sigmoid, relu, sin, identity) instead of a single fixed one. Second, it uses OriginConn, a connection gene that tracks historical innovation markers. When a node's activation is mutated, all connections touching that node receive new markers, which increases the genetic distance between the mutant and its parent. This pushes the mutant into a separate species, shielding it from crossover with genomes that still use the old activation.

The performance difference in Section 5.2 could come from either change, or from their interaction. To isolate the two, we ran an ablation that keeps the full HA-NEAT machinery (OriginConn, HANEATMutation, marker reassignment) but restricts it to a single activation function (tanh). The activation mutation still fires, but since there is only one option, nothing actually changes. This gives us HA-NEAT's structural changes without the diversity.

### Training performance

**Table 3. Ablation training results: aggregate normalized-minimum fitness.**
Pop=1024, 500 generations, 3 seeds. HA-NEAT ablation uses HA-NEAT infrastructure with tanh only.

| Method | Seed 42 | Seed 75 | Seed 123 | Mean |
|---|---|---|---|---|
| NEAT | 0.414 | 0.342 | 0.381 | 0.379 |
| HA-NEAT ablation (tanh only) | 0.352 | 0.342 | 0.398 | 0.364 |

<!-- Figure 6: Training curves for ablation.
     Source: text/figures/multi_task_neat_vs_haneat_ablation_training_*.pdf
     Three plots: Hopper, Walker2D, aggregated. Individual seed traces (dashed, low opacity) + bold mean. -->

The ablation averaged 0.364 across three seeds, slightly below NEAT's 0.379. The training curves (Figure 6) are more informative than the endpoint. On Hopper, both methods converge to similar levels (NEAT mean ~0.39, ablation ~0.34 normalized). Walker2D is where they diverge: NEAT pulls ahead after around generation 150 and ends at ~0.40 compared to the ablation's ~0.33. The aggregated fitness reflects that Walker2D gap, with NEAT at ~0.36 and the ablation at ~0.31 by generation 500.

### Re-evaluation

Training fitness comes from a single episode per genome, so we re-evaluated each of the six saved genomes over 30 episodes with different random seeds. Table 4 reports the results.

**Table 4. Ablation re-evaluation results: mean reward over 30 episodes per genome, 3 seeds per method.** Aggregate is the normalized minimum across both tasks, computed per seed and then averaged.

| Method | Hopper (mean +/- std) | Hopper (norm) | Walker2D (mean +/- std) | Walker2D (norm) | Aggregate (norm min) |
|---|---|---|---|---|---|
| NEAT | 1059 +/- 52 | 0.353 +/- 0.017 | 1288 +/- 635 | 0.258 +/- 0.127 | 0.241 +/- 0.101 |
| HA-NEAT ablation | 1083 +/- 76 | 0.361 +/- 0.025 | 1449 +/- 533 | 0.290 +/- 0.107 | 0.287 +/- 0.101 |

<!-- Figure 7: Re-evaluation bar chart.
     Source: text/figures/multi_task_neat_vs_haneat_ablation_evaluation_normalized.pdf
     Grouped bars per task, with seed-level means as scatter dots. -->

On re-evaluation the training-time gap narrows. The ablation scores slightly higher than NEAT on both individual tasks, and the aggregate normalized minimum tells the same story. But the seed-level variance is large, particularly on Walker2D where individual seed means range from 706 to 1966 for NEAT and from 932 to 1996 for the ablation. A difference of 0.046 against a shared standard deviation of 0.101 is not meaningful.

### Statistical comparison

With only three seeds per method, statistical power is limited. We report two non-parametric tests on seed-level means, plus effect sizes.

**Table 5. Statistical tests on seed-level means (n=3 vs n=3).**

| Task | Permutation test (p) | Mann-Whitney U (p) | Cohen's d | Interpretation |
|---|---|---|---|---|
| Hopper | 0.70 | 1.00 | -0.36 (small) | No significant difference |
| Walker2D | 0.70 | 0.70 | -0.28 (small) | No significant difference |

Neither task shows a significant difference (permutation p = 0.70 for both, Mann-Whitney p = 0.70-1.00). Effect sizes are small: Cohen's d of -0.36 on Hopper and -0.28 on Walker2D. One might be tempted to pool all 90 episodes per method and run a t-test, which does produce a lower p-value on Walker2D (Welch's t, p = 0.08). But episodes from the same seed share the same genome, so they are not independent observations. The proper unit of analysis is the seed-level mean, and at n = 3 the tests simply lack power to detect anything short of a very large effect.

Three seeds is a real limitation. Each ablation run takes [PLACEHOLDER: wall-clock time] on a single GPU, and scaling to 20-30 seeds was not feasible within our compute budget. Rather than relying on p-values that were never going to be informative at this sample size, we draw on three pieces of evidence together. The training curves show overlapping trajectories across 500 generations, not just similar endpoints. The re-evaluation effect sizes are small. And the direction of the (non-significant) difference actually favors the ablation slightly over NEAT, the opposite of what we would expect if HA-NEAT's machinery provided any advantage on its own.

Taken together, OriginConn and marker reassignment without activation diversity do not improve performance. The ablation slightly underperforms NEAT during training and performs comparably on re-evaluation. Activation diversity, not the surrounding infrastructure, appears to be what drives HA-NEAT's advantage in Section 5.2. [PLACEHOLDER: include full HA-NEAT bar from final experiment data for direct three-way comparison. Note the population size mismatch (ablation: pop 1024, main experiment: pop 4096) when interpreting magnitudes.]

---

## 5.7 Aggregation mode selection

The choice of normalized-minimum aggregation and the sweep that led to it are described in the Methodology (Section 3). In short, normalized-sum and normalized-product both allowed networks to specialize on one task while neglecting the other. Normalized-minimum forced balanced progress across both tasks, and all experiments reported above use it.

---
---

# Chapter 6: Discussion

## 6.1 Main finding

Across the experiments reported above, HA-NEAT with heterogeneous activations produced higher aggregate multi-task fitness than NEAT with uniform tanh. The ablation points to activation diversity as the source rather than the OriginConn/marker infrastructure. But the evidence has gaps: 2-3 seeds per condition in the main comparison, substantial episode-level variance in HA-NEAT genomes on re-evaluation, and an ablation run at a smaller population size than the main experiment. HA-NEAT being a better multi-task optimizer than NEAT is plausible from this data. It is not proven.

---

## 6.2 Activation diversity and multi-task generalization

The preliminary data suggest heterogeneous activations help, though the mechanism is not directly observable from fitness curves.

A plausible explanation is that activation diversity expands the function space accessible without topological changes. A NEAT network with only tanh activations must add nodes or connections to change its computational behavior. An HA-NEAT network can shift a node from saturating (tanh) to rectified (relu) without adding structure. In a multi-task setting where the same weights process 11-dimensional Hopper inputs and 17-dimensional Walker2D inputs, this allows the topology to be reused more efficiently across tasks.

Gaier and Ha (2019) showed in Weight Agnostic Neural Networks that varying the activation function globally across a fixed topology can solve diverse tasks, but they searched over activations at the architecture level, not per-node. Stanley and Miikkulainen's original NEAT (2002) used a single fixed activation (sigmoid or tanh). Per-node diversity in NEAT variants has been tried, but rarely on multi-task physical simulation problems. [PLACEHOLDER: add citation for any multi-task NEAT variants in the literature if they exist; check ALIFE/GECCO proceedings.]

A separate issue is the Walker2D variance. NEAT seed42's re-evaluation shows near-zero variance (std of 3-11 reward units), while HA-NEAT seed123 shows 435 on Hopper and 483 on Walker2D. HA-NEAT genomes may occupy a qualitatively different part of policy space, one that is more sensitive to initial conditions rather than strictly better. The training-best fitness comparison in Section 5.2 likely overstates HA-NEAT's advantage because it relies on the single best episode seen during training.

---

## 6.3 The role of speciation in HA-NEAT

When HA-NEAT mutates a node's activation, it reassigns historical markers on all connections touching that node. This pushes the mutant into its own species and shields it from crossover dilution. A side effect: HA-NEAT may produce more species than NEAT under identical settings. The v2 EDA found a positive correlation between species count and final fitness across NEAT runs, so if HA-NEAT's marker mechanism systematically inflates species count, the performance advantage attributed to activation diversity could be partially confounded by a speciation effect.

The ablation speaks to this. HA-NEAT machinery without activation diversity (Section 5.6) did not outperform NEAT; it slightly underperformed. If the marker mechanism alone were the driver, producing more species that explore more of the search space, we would expect the ablation to at least match NEAT. It did not. The diversity itself appears necessary. [PLACEHOLDER: check Figure 4 (species dynamics) to verify whether HA-NEAT ablation also produces more species than NEAT. If yes, and it doesn't help, this strengthens the activation-diversity interpretation.]

---

## 6.4 Training stochasticity and evaluation reliability

Section 5.4 exposed a failure mode worth discussing on its own: a genome that scores well on one episode may have gotten lucky rather than learned a robust policy. NEAT seed42's near-zero re-evaluation variance is consistent with a stable, somewhat stereotyped gait. It always does roughly the same thing. HA-NEAT seed42's moderate variance suggests a policy that is more sensitive to initial conditions, sometimes doing better and sometimes doing worse.

This matters for the performance claims. The MLflow `best_fitness` in Table 1 is the single best episode seen during training. For HA-NEAT seed42 in v4, the training-best Walker2D was 2365, but the 5-episode re-evaluation gave 2005 ± 162, a 15% drop. For HA-NEAT seed123, Hopper went from 1370 to 752, a 45% drop. Any final comparison between methods needs multi-episode re-evaluation, not just the training checkpoint number.

---

## 6.5 Limitations

The main comparisons in Sections 5.2–5.3 rest on 2–3 seeds per condition. With this sample size it is not possible to compute meaningful confidence intervals or run significance tests. The final experiment uses 5 seeds, which is better but still at the low end for detecting moderate effect sizes — any performance difference below ~0.05 normalized fitness units should be treated as noise given the observed cross-seed variance.

Both NEAT and HA-NEAT used the same `compatibility_threshold`, `max_stagnation`, `survival_threshold`, and population size. HA-NEAT introduces `activation_mutate_rate` (fixed at 0.1 across all runs), which was not swept in the final experiment. It is possible that NEAT would benefit from different speciation settings than HA-NEAT, or that HA-NEAT's activation mutation rate is suboptimal. The comparison is between default-configured methods, not optimally-configured ones.

Hopper and Walker2D are both bipedal locomotion tasks on flat terrain with the same reward function structure (forward velocity + control cost + alive bonus) and the same physics backend. They differ in dimensionality and difficulty, but they are not qualitatively different problem types. Generalization claims do not extend beyond this task family; it is unknown whether the findings hold for tasks with different objectives (manipulation, navigation) or substantially different physics.

The fitness estimate used during training is a single episode per genome. This affects both methods, but Section 5.4 shows it may affect HA-NEAT more if HA-NEAT genomes have higher episode variance. The training dynamics in Figures 1–2 reflect this noisy signal, not true expected policy performance.

Finally, the experiment went through several iterations (v1 through final) as early results revealed problems: floor-penetration exploits in the generalized backend, aggregation mode imbalance, and under-powered population sizes and generation counts. The hypothesis was refined alongside the data. This is common in empirical ML work, but it means the results should be read as exploratory rather than confirmatory.

---

## 6.6 Future work

The most immediate next step is more environments. Ant adds 3D movement, HalfCheetah is a faster locomotion task with simpler contacts, and Humanoid is high-dimensional and unstable. Running the same comparison on 3 or 4 tasks would test whether the activation diversity advantage scales or was specific to the Hopper/Walker2D pair.

More seeds would also help. With 10-20 seeds per condition, Wilcoxon signed-rank tests and effect sizes with confidence intervals become feasible. Five seeds is enough to see trends but not enough to make strong claims.

HA-NEAT's `activation_mutate_rate` was fixed at 0.1 without sweeping. A comparison of {0.05, 0.10, 0.20} would test whether 0.10 is actually a good value or just an arbitrary default. NEAT may also need a different `compatibility_threshold` than HA-NEAT to produce comparable species counts.

The `.npz` checkpoints contain the per-node activation assignments for every trained HA-NEAT genome, but we have not analyzed them. Do certain activations cluster near input nodes? Do genomes that score well on Walker2D favor relu over tanh in specific positions? This kind of analysis would add mechanistic evidence to what is currently just a performance comparison.

A comparison against gradient-based multi-task RL (multi-task SAC, MTRL variants) would put the neuroevolution numbers in a broader context. Gradient methods are not constrained to a single fixed-topology network, so the comparison is not apples-to-apples, but knowing the rough performance gap would be useful.

---

## 6.7 Closing

HA-NEAT extends NEAT with per-node activation diversity and a speciation mechanism that protects activation-mutant genomes. On a two-task Brax locomotion problem, the preliminary evidence suggests activation diversity provides a measurable benefit over uniform tanh, and the ablation puts the source of that benefit on the diversity rather than the OriginConn machinery. The final experiment will confirm or complicate this picture. But regardless of how that comparison resolves, the re-evaluation results are the more general finding: single-episode training fitness in NEAT-based RL is a noisy proxy, and performance numbers reported from a single saved checkpoint should always be accompanied by multi-episode evaluation before any claim is made.
