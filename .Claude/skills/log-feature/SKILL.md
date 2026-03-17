---
name: log-feature
description: Use when a new feature has been implemented and is ready to be logged — after implementation is complete and verified, before committing or moving on.
---

# Log Feature

## Overview

Every completed feature gets a structured entry appended to `docs/FEATURE_LOG.md`. This creates a running record of what was built, why, and how it fits the thesis experiment.

## When to Use

- A new algorithm, component, or behavior has been implemented and verified
- A significant change to existing functionality is complete
- Before committing a feature branch

## Steps

1. **Append an entry** to `docs/FEATURE_LOG.md` using the template below
2. **Stage the log file** alongside the feature commit

## Entry Template

```markdown
## [YYYY-MM-DD] <Feature Name>

**Branch:** `feat/...`
**Status:** Complete

### What
One sentence: what was built.

### Why
One sentence: why it matters for the NEAT vs HA-NEAT comparison.

### Key files changed
- `path/to/file.py` — what changed

### Notes
Any gotchas, design decisions, or follow-up items.
```

## Example

```markdown
## [2026-03-12] NormalizedWeightedSum Aggregator

**Branch:** `feat/normalized-fitness-aggregation`
**Status:** Complete

### What
Added `NormalizedWeightedSum` aggregator that divides per-task fitness by `max_reward` before weighting, auto-selected when `TaskSpec.max_reward != 1.0`.

### Why
Hopper (max ~3000) and Walker2D (max ~5000) have different reward scales — plain weighted sum would bias toward Walker2D.

### Key files changed
- `src/tensorneat/problem/rl/multi_task.py` — added aggregator + auto-selection logic
- `test/test_multi_task_aggregation.py` — unit tests

### Notes
`BRAX_REFERENCE_REWARDS` dict provides canonical max rewards for known envs.
```
