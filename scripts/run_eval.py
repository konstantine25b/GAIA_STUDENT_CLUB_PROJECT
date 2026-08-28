"""CLI entry point for full model evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import setup_project

setup_project()

from src.dataset import load_models
from src.evaluator import evaluate_model
from src.paths import MODELS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all models on the dataset with n trials per question."
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
        help="Optional run folder name under results/runs/<model>/",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else load_models(MODELS_FILE)
    for model in models:
        print(f"\n=== Starting {model} ===")
        paths = evaluate_model(
            model,
            n_trials=args.n_trials,
            run_id=args.run_id,
            limit=args.limit,
        )
        print(f"Results: {paths.results_csv}")
        print(f"Raw log: {paths.raw_jsonl}")


if __name__ == "__main__":
    main()
