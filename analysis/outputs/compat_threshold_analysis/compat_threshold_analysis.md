# Compatibility threshold analysis — multi_task_neat_vs_haneat_final

5 seeds per condition, 2000 generations, pop 4096, normalized_min on Hopper + Walker2D.

## Raw results (best fitness per run)

| Algorithm | Compat | Seed 42 | Seed 123 | Seed 75 | Seed 7 | Seed 21 | Mean |
|---|---|---|---|---|---|---|---|
| NEAT | 0.3 | 0.521 | 0.492 | 0.342 | 0.538 | 0.475 | 0.474 |
| NEAT | 0.5 | 0.433 | 0.354 | 0.401 | 0.417 | 0.397 | 0.400 |
| HA-NEAT | 0.3 | 0.437 | 0.410 | 0.361 | 0.458 | 0.392 | 0.412 |
| HA-NEAT | 0.5 | 0.505 | 0.457 | 0.462 | 0.388 | 0.460 | 0.454 |

## Key finding

NEAT and HA-NEAT prefer different thresholds. NEAT at 0.3 outperforms 0.5 by +0.074 mean fitness (+18.5%). HA-NEAT flips this: 0.5 beats 0.3 by +0.042 (+10.2%).

This is the awkward result. It means there's no single threshold that's optimal for both — which rules out a simple "pick the winner" hyperparameter decision.

## Saturation curve (NEAT compat 0.3, representative seeds)

| Generation | s1 | s2 | s3 | s5 |
|---|---|---|---|---|
| 100 | 0.340 | 0.248 | 0.280 | 0.295 |
| 300 | 0.391 | 0.382 | 0.312 | 0.332 |
| 500 | 0.410 | 0.391 | 0.391 | 0.339 |
| 750 | 0.466 | 0.396 | 0.437 | 0.340 |
| 1000 | 0.420 | 0.396 | 0.428 | 0.316 |
| 1500 | 0.430 | 0.455 | 0.441 | 0.256 |
| 2000 | 0.432 | 0.472 | 0.436 | 0.251 |

Two phases: rapid rise through gen 300, plateau to ~700, then a second jump in some seeds around 700–1200. Beyond gen 1200 there's no consistent improvement — and s5 actually declines after gen 1000.

## Decision for reduced experiment

Using compat 0.3 for both algorithms. The alternative — per-algorithm optimal thresholds — would confound speciation pressure with activation diversity. A reviewer could argue HA-NEAT just benefits from larger species, not from diverse activations. Compat 0.3 is NEAT's stronger setting, so an HA-NEAT win there is harder to dismiss on methodological grounds.

The differential sensitivity is worth a sentence in the discussion but shouldn't drive the experimental design.

Generation limit: 750. Covers both saturation phases, cuts compute by 62.5% vs the 2000-gen runs.
