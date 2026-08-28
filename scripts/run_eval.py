"""CLI entry point for full model evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import setup_project

setup_project()

from src.dataset import load_models
from src.evaluator import evaluate_run
from src.paths import MODELS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation trial-by-trial: for each trial, every model answers "
            "all questions in parallel, saving progress after each API call."
        )
    )
    parser.add_argument(
        "--model",
        help="Run a single model (default: all models in config/models.txt)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override config/inference.json n_trials",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions (for dry runs)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Folder name under results/runs/ (required with --continue if multiple runs)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Resume the latest run (or --run-id). Skips completed trials.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel API threads (default: config inference.json max_workers)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else load_models(MODELS_FILE)
    paths = evaluate_run(
        models,
        n_trials=args.n_trials,
        run_id=args.run_id,
        limit=args.limit,
        continue_run=args.continue_run,
        max_workers=args.workers,
    )
    print(f"\nRun directory: {paths.run_dir}")
    print(f"Results: {paths.results_csv}")
    print(f"Raw responses: {paths.raw_responses_csv}")


if __name__ == "__main__":
    main()
