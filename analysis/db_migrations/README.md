# MLflow database migrations

One-off scripts that mutate `mlflow.db` to recover from interrupted training
runs or merge runs across experiment IDs. **These are not part of the
experimental pipeline** — they were applied once to produce the canonical
analysis dataset and should not be re-run on a fresh database.

## Scripts

### `merge_neat_haneat_experiments.py`

Combines NEAT runs from `multi_task_neat_vs_haneat_final` (MLflow exp 9)
with HA-NEAT runs from `multi_task_haneat_finalv2` (exp 11) into a new
experiment `multi_task_final_combined`.

**Why:** The `final` experiment was split across two GPUs because a single
run aborted mid-way. The HA-NEAT side was restarted as `finalv2` on the
second GPU. This script unified the surviving FINISHED runs into one
queryable experiment for analysis.

Hardcoded paths assume the script runs on the GPU server at
`/home/20240503/tensorneat`.
