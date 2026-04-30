"""
NEAT vs HA-NEAT ablation analysis pipeline.

Compares standard NEAT against HA-NEAT restricted to tanh-only activations,
testing whether activation diversity is the active ingredient in HA-NEAT.

Steps:
  1. Re-evaluate all genomes (n_eval episodes/task, MJX backend)
  2. Plot evaluation bar charts
  3. Run statistical tests → write txt + json

Usage:
    # Full run (re-eval + plots + stats):
    uv run python -m analysis.run_ablation_analysis

    # Skip re-eval if combined.json already exists:
    uv run python -m analysis.run_ablation_analysis --skip-reeval

    # Smoke test (3 episodes):
    uv run python -m analysis.run_ablation_analysis --n-eval 3
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "text", "figures"))

EXPERIMENT = "multi_task_neat_vs_haneat_ablation"
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
    out = os.path.join(output_root, f"{EXPERIMENT}_{ts}")
    for sub in ["reeval", "eval_plots", "stats"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Step 1 – Re-evaluate
# ---------------------------------------------------------------------------

def step_reevaluate(project_root: str, out_dir: str, n_eval: int, force: bool) -> dict:
    from analysis.merge import collect_genomes
    from analysis.reevaluate import run_reevaluation, build_combined_json, save_json

    combined_path = os.path.join(out_dir, "reeval", "combined.json")
    if os.path.exists(combined_path) and not force:
        logging.info("=== STEP 1: Skipping re-evaluation (combined.json exists) ===")
        with open(combined_path) as f:
            return json.load(f)

    logging.info("=== STEP 1: Collecting genomes ===")
    results_root = os.path.join(project_root, "results")
    entries = collect_genomes(results_root, [EXPERIMENT])

    neat_entries = [e for e in entries if e.algorithm == "neat"]
    ablation_entries = [e for e in entries if e.algorithm == "ha_neat_ablation"]
    logging.info(f"  NEAT: {len(neat_entries)} genomes")
    logging.info(f"  HA-NEAT ablation: {len(ablation_entries)} genomes")

    logging.info(f"=== Re-evaluating {len(entries)} genomes ({n_eval} eps/task) ===")
    results = run_reevaluation(entries, n_eval=n_eval, backend=BACKEND)
    combined = build_combined_json(results, n_eval=n_eval, backend=BACKEND, experiments=[EXPERIMENT])
    save_json(combined, combined_path)
    logging.info(f"  Saved: {combined_path}")
    return combined


# ---------------------------------------------------------------------------
# Step 2 – Evaluation bar plots
# ---------------------------------------------------------------------------

def step_eval_plots(combined: dict, out_dir: str, fmt: str) -> None:
    from plot_evaluation_results import setup_matplotlib, plot_evaluation

    logging.info("=== STEP 2: Plotting evaluation bar charts ===")
    plots_dir = os.path.join(out_dir, "eval_plots")
    setup_matplotlib()

    # Remap ha_neat_ablation → ha_neat so plot_evaluation renders it correctly
    combined_remapped = dict(combined)
    combined_remapped["runs"] = [
        dict(r, algorithm="ha_neat") if r.get("algorithm") == "ha_neat_ablation" else r
        for r in combined["runs"]
    ]
    summary_remapped: dict = {}
    if "neat" in combined.get("summary", {}):
        summary_remapped["neat"] = combined["summary"]["neat"]
    if "ha_neat_ablation" in combined.get("summary", {}):
        summary_remapped["ha_neat"] = combined["summary"]["ha_neat_ablation"]
    combined_remapped["summary"] = summary_remapped
    combined_remapped["metadata"] = dict(combined["metadata"], experiment=EXPERIMENT)

    plot_evaluation(combined_remapped, TASKS, plots_dir, fmt, normalized=False)
    plot_evaluation(combined_remapped, TASKS, plots_dir, fmt, normalized=True)
    logging.info("Evaluation plots done.")


# ---------------------------------------------------------------------------
# Step 3 – Statistical tests
# ---------------------------------------------------------------------------

def step_stats(combined: dict, out_dir: str) -> dict:
    from analysis.stats import compute_tests, write_stats

    logging.info("=== STEP 3: Running statistical tests ===")
    runs_ok = [r for r in combined["runs"] if "error" not in r]
    neat_runs = [r for r in runs_ok if r.get("algorithm") == "neat"]
    ablation_runs = [r for r in runs_ok if r.get("algorithm") == "ha_neat_ablation"]

    logging.info(f"  NEAT: {len(neat_runs)} seeds | HA-NEAT ablation: {len(ablation_runs)} seeds")
    test_results = compute_tests(neat_runs, ablation_runs, label_a="neat", label_b="ha_neat_ablation")
    write_stats(test_results, os.path.join(out_dir, "stats"), label_a="NEAT", label_b="HA-NEAT ablation")
    logging.info("Stats done.")
    return test_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEAT vs HA-NEAT ablation analysis pipeline.")
    p.add_argument("--skip-reeval", action="store_true", help="Skip re-evaluation if combined.json exists")
    p.add_argument("--force-reeval", action="store_true", help="Force re-evaluation even if combined.json exists")
    p.add_argument("--n-eval", type=int, default=30, help="Episodes per genome per task (default: 30)")
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
    logging.info(f"Experiment: {EXPERIMENT}")

    combined = step_reevaluate(
        PROJECT_ROOT, out_dir, args.n_eval,
        force=args.force_reeval and not args.skip_reeval,
    )

    step_eval_plots(combined, out_dir, args.fmt)
    step_stats(combined, out_dir)

    logging.info(f"\n=== DONE — results in {out_dir} ===")


if __name__ == "__main__":
    main()
