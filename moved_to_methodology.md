# Content removed from Chapter 5 (Results) — to be placed in Methodology

- Reference rewards used for normalization: 3000 (Hopper), 5000 (Walker2D), 6000 (Ant), 8000 (HalfCheetah), 6000 (Humanoid), 5 (Reacher). Drawn from `BRAX_REFERENCE_REWARDS`.
- Aggregation mode selection and justification (original Section 5.7 content):
  - Three modes tested in initial sweep (36 runs, pop {512, 1024, 2048}, gen {100, 200}, seed 123): `normalized_sum`, `normalized_product`, `normalized_min`
  - `normalized_product` produced highest single-task score (Walker2D=3606, 0.721 norm) but Hopper was only 1096 (0.365 norm) — genome exploited one task
  - `normalized_sum` showed similar one-sided specialization
  - `normalized_min` forced balanced progress across both tasks; best run scored Hopper=1182, Walker2D=2012
  - All subsequent experiments used `normalized_min`
  - Source: MLflow experiment `multi_task_neat_vs_haneat_mjx_backend`
  - Figure showing aggregation mode comparison: two panels (Hopper, Walker2D), three lines per panel (normalized_sum, normalized_min, normalized_product), pop=1024, gen=200
