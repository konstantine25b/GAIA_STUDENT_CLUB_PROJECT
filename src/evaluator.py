"""Run models on the dataset and write per-question trial results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.dataset import format_prompt, load_dataset
from src.llm_client import chat_with_retry, load_inference_config
from src.parser import extract_answer_from_row
from src.paths import RESULTS_DIR

RAW_RESPONSE_COLUMNS = [
    "trial",
    "llm_model",
    "question_id",
    "question_category",
    "correct_answer",
    "prompt",
    "raw_output",
    "parsed_answer",
]


@dataclass
class RunPaths:
    run_dir: Path
    results_csv: Path
    raw_responses_csv: Path
    metadata_json: Path


def _trial_columns(n_trials: int) -> list[str]:
    return [f"it_{i}_ans" for i in range(1, n_trials + 1)]


def results_columns(n_trials: int) -> list[str]:
    return [
        "llm_model",
        "question_category",
        "question_id",
        "options",
        "correct_answer",
        *_trial_columns(n_trials),
        "correct_answered_num",
        "accuracy",
    ]


def get_run_paths(run_id: str | None = None) -> RunPaths:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        results_csv=run_dir / "results.csv",
        raw_responses_csv=run_dir / "raw_responses.csv",
        metadata_json=run_dir / "run_metadata.json",
    )


def _result_key(model: str, question_id: int) -> tuple[str, int]:
    return model, question_id


def _empty_result_row(model: str, row: pd.Series, n_trials: int) -> dict:
    gold = str(row["answer"]).strip().upper()
    result = {
        "llm_model": model,
        "question_category": row["category"],
        "question_id": int(row["question_id"]),
        "options": row["options"],
        "correct_answer": gold,
        "correct_answered_num": 0,
        "accuracy": 0.0,
    }
    for i in range(1, n_trials + 1):
        result[f"it_{i}_ans"] = ""
    return result


def _finalize_result_row(row: dict, n_trials: int) -> dict:
    gold = row["correct_answer"]
    answers = [row.get(f"it_{i}_ans") or "" for i in range(1, n_trials + 1)]
    correct_count = sum(1 for ans in answers if ans == gold)
    row["correct_answered_num"] = correct_count
    row["accuracy"] = correct_count / n_trials
    return row


def _write_results_csv(results_csv: Path, rows: list[dict], n_trials: int) -> None:
    df = pd.DataFrame(rows, columns=results_columns(n_trials))
    df = df.sort_values(["llm_model", "question_id"]).reset_index(drop=True)
    df.to_csv(results_csv, index=False)


def _append_raw_responses_csv(raw_csv: Path, records: list[dict]) -> None:
    if not records:
        return
    frame = pd.DataFrame(records, columns=RAW_RESPONSE_COLUMNS)
    write_header = not raw_csv.exists()
    frame.to_csv(raw_csv, mode="a", header=write_header, index=False)


def _load_existing_raw_keys(raw_csv: Path) -> set[tuple[int, str, int]]:
    if not raw_csv.exists():
        return set()
    df = pd.read_csv(raw_csv)
    return {
        (int(row.trial), str(row.llm_model), int(row.question_id))
        for row in df.itertuples(index=False)
    }


def _load_results_state(
    results_csv: Path,
    models: list[str],
    df: pd.DataFrame,
    n_trials: int,
) -> dict[tuple[str, int], dict]:
    state: dict[tuple[str, int], dict] = {}
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        for record in existing.to_dict(orient="records"):
            key = _result_key(str(record["llm_model"]), int(record["question_id"]))
            state[key] = record

    for model in models:
        for _, row in df.iterrows():
            key = _result_key(model, int(row["question_id"]))
            if key not in state:
                state[key] = _empty_result_row(model, row, n_trials)
    return state


def evaluate_run(
    models: list[str],
    *,
    dataset: pd.DataFrame | None = None,
    n_trials: int | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> RunPaths:
    """Run trial-by-trial: each trial runs every model on every question."""
    cfg = load_inference_config()
    trials = n_trials if n_trials is not None else int(cfg["n_trials"])
    df = dataset if dataset is not None else load_dataset()
    if limit is not None:
        df = df.head(limit)

    paths = get_run_paths(run_id=run_id)
    done_keys = _load_existing_raw_keys(paths.raw_responses_csv)
    results_state = _load_results_state(paths.results_csv, models, df, trials)

    if not paths.metadata_json.exists():
        paths.metadata_json.write_text(
            json.dumps(
                {
                    "models": models,
                    "n_trials": trials,
                    "inference_config": cfg,
                    "dataset_rows": len(df),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    total_questions = len(df)
    total_models = len(models)

    for trial in range(1, trials + 1):
        trial_records: list[dict] = []
        print(f"\n=== Trial {trial}/{trials} ===")

        for model_idx, model in enumerate(models, start=1):
            print(f"\n--- Model {model_idx}/{total_models}: {model} ---")
            for q_idx, (_, row) in enumerate(df.iterrows(), start=1):
                qid = int(row["question_id"])
                key = (trial, model, qid)
                if key in done_keys:
                    continue

                prompt = format_prompt(row)
                print(
                    f"[trial {trial}/{trials}] [{model}] "
                    f"question {q_idx}/{total_questions} (id={qid})"
                )
                output = chat_with_retry(prompt, model=model)
                pred = extract_answer_from_row(output, row["options"]) or ""

                trial_records.append(
                    {
                        "trial": trial,
                        "llm_model": model,
                        "question_id": qid,
                        "question_category": row["category"],
                        "correct_answer": str(row["answer"]).strip().upper(),
                        "prompt": prompt,
                        "raw_output": output,
                        "parsed_answer": pred,
                    }
                )

                result_key = _result_key(model, qid)
                results_state[result_key][f"it_{trial}_ans"] = pred
                done_keys.add(key)

        _append_raw_responses_csv(paths.raw_responses_csv, trial_records)
        rows = [
            _finalize_result_row(dict(row), trials)
            for row in results_state.values()
        ]
        _write_results_csv(paths.results_csv, rows, trials)
        print(f"Saved trial {trial} raw rows: {len(trial_records)}")
        print(f"Updated: {paths.raw_responses_csv}")
        print(f"Updated: {paths.results_csv}")

    return paths
