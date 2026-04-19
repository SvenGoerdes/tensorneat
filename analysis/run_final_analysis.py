"""
Final NEAT vs HA-NEAT analysis pipeline.

Steps:
  1. (Optional) Sync mlflow.db + .npz results from remote server
  2. Merge + deduplicate HA-NEAT genomes from both experiments
  3. Re-evaluate all genomes (20 episodes/task, MJX backend)
  4. Plot training curves from MLflow
  5. Plot evaluation bar charts
  6. Run statistical tests → write txt + json
  7. Generate markdown interpretation

Usage:
    # Full run (sync + re-eval):
    TENSORNEAT_SSH_PASS='...' uv run python -m analysis.run_final_analysis

    # Skip sync (files already local):
    uv run python -m analysis.run_final_analysis --skip-sync

    # Smoke test (3 episodes, local files):
    uv run python -m analysis.run_final_analysis --skip-sync --n-eval 3

    # Skip re-eval if combined.json already exists:
    uv run python -m analysis.run_final_analysis --skip-sync --skip-reeval
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "text", "figures"))

EXPERIMENTS = [
    "multi_task_neat_vs_haneat_final",
    "multi_task_haneat_finalv2",
]
TASKS = ["hopper", "walker2d"]
BACKEND = "mjx"


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )


def make_output_dir(output_root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_root, f"final_{ts}")
    for sub in ["reeval", "training_curves", "eval_plots", "stats"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Step 1 – Sync
# ---------------------------------------------------------------------------

def step_sync(project_root: str) -> None:
    from analysis.sync import sync_db, sync_results
    logging.info("=== STEP 1: Syncing from server ===")
    sync_db(os.path.join(project_root, "mlflow.db"))
    sync_results(os.path.join(project_root, "results"))
    logging.info("Sync complete.")


# ---------------------------------------------------------------------------
# Step 2+3 – Merge + Re-evaluate
# ---------------------------------------------------------------------------

def step_reevaluate(project_root: str, out_dir: str, n_eval: int, force: bool) -> dict:
    from analysis.merge import collect_genomes, dedup_haneat
    from analysis.reevaluate import run_reevaluation, build_combined_json, save_json

    combined_path = os.path.join(out_dir, "reeval", "combined.json")
    if os.path.exists(combined_path) and not force:
        logging.info(f"=== STEP 2+3: Skipping re-evaluation (combined.json exists) ===")
        with open(combined_path) as f:
            return json.load(f)

    logging.info("=== STEP 2: Collecting + merging genomes ===")
    results_root = os.path.join(project_root, "results")
    all_entries = collect_genomes(results_root, EXPERIMENTS)
    deduped = dedup_haneat(all_entries)

    neat_entries = [e for e in deduped if e.algorithm == "neat"]
    ha_entries = [e for e in deduped if e.algorithm == "ha_neat"]
    logging.info(f"  NEAT: {len(neat_entries)} genomes")
    logging.info(f"  HA-NEAT: {len(ha_entries)} genomes (after dedup)")

    logging.info(f"=== STEP 3: Re-evaluating {len(deduped)} genomes ({n_eval} eps/task) ===")
    results = run_reevaluation(deduped, n_eval=n_eval, backend=BACKEND)
    combined = build_combined_json(results, n_eval=n_eval, backend=BACKEND, experiments=EXPERIMENTS)
    save_json(combined, combined_path)
    return combined


# ---------------------------------------------------------------------------
# Step 4 – Training curves
# ---------------------------------------------------------------------------

def step_training_curves(project_root: str, out_dir: str, fmt: str) -> None:
    from plot_training_curves import load_metrics, setup_matplotlib, plot_per_task, plot_aggregated

    logging.info("=== STEP 4: Plotting training curves ===")
    db_path = os.path.join(project_root, "mlflow.db")
    curves_dir = os.path.join(out_dir, "training_curves")
    metric_keys = [f"best_fitness_normalized/{t}" for t in TASKS] + ["fitness/max"]

    setup_matplotlib()

    # Both experiments share the same run names; load and merge by combining dicts
    merged_data: dict = {}
    for exp in EXPERIMENTS:
        try:
            data = load_metrics(db_path, exp, metric_keys)
            merged_data.update(data)
            logging.info(f"  Loaded {len(data)} runs from '{exp}'")
        except SystemExit:
            logging.warning(f"  Could not load experiment '{exp}' from MLflow — skipping curves for it.")

    if not merged_data:
        logging.warning("  No MLflow data loaded — skipping training curves.")
        return

    for task in TASKS:
        plot_per_task(merged_data, task, "best_fitness_normalized", "final_combined", curves_dir, fmt)
    plot_aggregated(merged_data, "final_combined", curves_dir, fmt)
    logging.info("Training curves done.")


# ---------------------------------------------------------------------------
# Step 5 – Evaluation bar plots
# ---------------------------------------------------------------------------

def step_eval_plots(combined: dict, out_dir: str, fmt: str) -> None:
    from plot_evaluation_results import setup_matplotlib, plot_evaluation

    logging.info("=== STEP 5: Plotting evaluation bar charts ===")
    plots_dir = os.path.join(out_dir, "eval_plots")
    setup_matplotlib()

    # Override experiment name so filenames are clean
    combined_copy = dict(combined)
    combined_copy["metadata"] = dict(combined["metadata"])
    combined_copy["metadata"]["experiment"] = "final_combined"

    plot_evaluation(combined_copy, TASKS, plots_dir, fmt, normalized=False)
    plot_evaluation(combined_copy, TASKS, plots_dir, fmt, normalized=True)
    logging.info("Evaluation plots done.")


# ---------------------------------------------------------------------------
# Step 6 – Statistical tests
# ---------------------------------------------------------------------------

def step_stats(combined: dict, out_dir: str) -> dict:
    from analysis.stats import compute_tests, write_stats

    logging.info("=== STEP 6: Running statistical tests ===")
    runs_ok = [r for r in combined["runs"] if "error" not in r]
    neat_runs = [r for r in runs_ok if r.get("algorithm") == "neat"]
    ha_runs = [r for r in runs_ok if r.get("algorithm") == "ha_neat"]

    logging.info(f"  NEAT: {len(neat_runs)} seeds | HA-NEAT: {len(ha_runs)} seeds")
    test_results = compute_tests(neat_runs, ha_runs, label_a="neat", label_b="ha_neat")
    write_stats(test_results, os.path.join(out_dir, "stats"), label_a="NEAT", label_b="HA-NEAT")
    logging.info("Stats done.")
    return test_results


# ---------------------------------------------------------------------------
# Step 7 – Interpretation markdown
# ---------------------------------------------------------------------------

def step_interpretation(combined: dict, test_results: dict, out_dir: str, n_eval: int) -> None:
    from analysis.interpretation import generate_markdown, write_interpretation

    logging.info("=== STEP 7: Generating markdown interpretation ===")
    runs_ok = [r for r in combined["runs"] if "error" not in r]
    ha_provenance = [
        {"file": r["file"], "experiment": r.get("experiment", "?"),
         "seed": r.get("seed"), "compat": r.get("compat")}
        for r in runs_ok if r.get("algorithm") == "ha_neat"
    ]

    md = generate_markdown(
        test_results=test_results,
        combined_json=combined,
        label_a="neat",
        label_b="ha_neat",
        n_eval=n_eval,
        backend=BACKEND,
        provenance=ha_provenance,
    )
    write_interpretation(md, out_dir)
    logging.info("Interpretation done.")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(combined: dict, out_dir: str, args: argparse.Namespace) -> None:
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    runs_ok = [r for r in combined["runs"] if "error" not in r]
    meta = {
        "git_sha": git_sha,
        "timestamp": datetime.now().isoformat(),
        "n_eval": args.n_eval,
        "backend": BACKEND,
        "experiments": EXPERIMENTS,
        "n_neat": sum(1 for r in runs_ok if r.get("algorithm") == "neat"),
        "n_haneat": sum(1 for r in runs_ok if r.get("algorithm") == "ha_neat"),
        "skip_sync": args.skip_sync,
        "force_reeval": args.force_reeval,
    }
    path = os.path.join(out_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final NEAT vs HA-NEAT analysis pipeline.")
    p.add_argument("--skip-sync", action="store_true", help="Skip server sync step")
    p.add_argument("--skip-reeval", action="store_true", help="Skip re-evaluation if combined.json exists")
    p.add_argument("--force-reeval", action="store_true", help="Force re-evaluation even if combined.json exists")
    p.add_argument("--n-eval", type=int, default=20, help="Episodes per genome per task")
    p.add_argument("--output-root", default=os.path.join(PROJECT_ROOT, "analysis", "outputs"),
                   help="Root directory for output folders")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], dest="fmt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    out_dir = make_output_dir(args.output_root)

    setup_logging(os.path.join(out_dir, "log.txt"))
    logging.info(f"Output directory: {out_dir}")

    if not args.skip_sync:
        step_sync(PROJECT_ROOT)

    skip_reeval = args.skip_reeval and not args.force_reeval
    combined = step_reevaluate(PROJECT_ROOT, out_dir, args.n_eval, force=args.force_reeval)

    step_training_curves(PROJECT_ROOT, out_dir, args.fmt)
    step_eval_plots(combined, out_dir, args.fmt)
    test_results = step_stats(combined, out_dir)
    step_interpretation(combined, test_results, out_dir, args.n_eval)
    write_metadata(combined, out_dir, args)

    logging.info(f"\n=== DONE — results in {out_dir} ===")


if __name__ == "__main__":
    main()
