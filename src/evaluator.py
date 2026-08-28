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


@dataclass
class EvalPaths:
    model: str
    run_dir: Path
    results_csv: Path
    raw_jsonl: Path
    metadata_json: Path


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def get_eval_paths(model: str, run_id: str | None = None) -> EvalPaths:
    slug = model_slug(model)
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / slug / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return EvalPaths(
        model=model,
        run_dir=run_dir,
        results_csv=run_dir / "results.csv",
        raw_jsonl=run_dir / "raw_responses.jsonl",
        metadata_json=run_dir / "run_metadata.json",
    )


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


def _load_completed_question_ids(raw_jsonl: Path) -> set[int]:
    if not raw_jsonl.exists():
        return set()
    done: set[int] = set()
    with raw_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete":
                done.add(int(record["question_id"]))
    return done


def _append_raw_record(raw_jsonl: Path, record: dict) -> None:
    with raw_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_result_row(
    model: str,
    row: pd.Series,
    trial_answers: list[str | None],
    n_trials: int,
) -> dict:
    gold = str(row["answer"]).strip().upper()
    parsed = [ans for ans in trial_answers if ans]
    correct_count = sum(1 for ans in trial_answers if ans == gold)
    result = {
        "llm_model": model,
        "question_category": row["category"],
        "question_id": int(row["question_id"]),
        "options": row["options"],
        "correct_answer": gold,
        "correct_answered_num": correct_count,
        "accuracy": correct_count / n_trials,
    }
    for i in range(1, n_trials + 1):
        ans = trial_answers[i - 1]
        result[f"it_{i}_ans"] = ans if ans else ""
    return result


def _write_results_csv(results_csv: Path, rows: list[dict], n_trials: int) -> None:
    df = pd.DataFrame(rows, columns=results_columns(n_trials))
    df.to_csv(results_csv, index=False)


def evaluate_model(
    model: str,
    *,
    dataset: pd.DataFrame | None = None,
    n_trials: int | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> EvalPaths:
    cfg = load_inference_config()
    trials = n_trials if n_trials is not None else int(cfg["n_trials"])
    df = dataset if dataset is not None else load_dataset()
    if limit is not None:
        df = df.head(limit)

    paths = get_eval_paths(model, run_id=run_id)
    completed = _load_completed_question_ids(paths.raw_jsonl)
    rows: list[dict] = []

    if paths.results_csv.exists():
        existing = pd.read_csv(paths.results_csv)
        rows = existing.to_dict(orient="records")

    if not paths.metadata_json.exists():
        paths.metadata_json.write_text(
            json.dumps(
                {
                    "model": model,
                    "n_trials": trials,
                    "inference_config": cfg,
                    "dataset_rows": len(df),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    total = len(df)
    for idx, row in df.iterrows():
        qid = int(row["question_id"])
        if qid in completed:
            continue

        prompt = format_prompt(row)
        trial_answers: list[str | None] = []

        for trial in range(1, trials + 1):
            print(
                f"[{model}] question {len(completed) + 1}/{total} "
                f"(id={qid}) trial {trial}/{trials}"
            )
            output = chat_with_retry(prompt, model=model)
            pred = extract_answer_from_row(output, row["options"])
            trial_answers.append(pred)
            trial_record: dict = {
                "model": model,
                "question_id": qid,
                "trial": trial,
                "parsed_answer": pred,
            }
            if cfg.get("save_full_raw_output", False):
                trial_record["raw_output"] = output
            else:
                trial_record["output_tail"] = output[-300:]
            _append_raw_record(paths.raw_jsonl, trial_record)

        result_row = _build_result_row(model, row, trial_answers, trials)
        rows.append(result_row)
        _write_results_csv(paths.results_csv, rows, trials)
        _append_raw_record(
            paths.raw_jsonl,
            {
                "model": model,
                "question_id": qid,
                "status": "complete",
                "trial_answers": trial_answers,
            },
        )
        completed.add(qid)

    return paths
