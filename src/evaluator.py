"""Run models on the dataset and write per-question trial results."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass(frozen=True)
class EvalTask:
    trial: int
    model: str
    question_id: int
    category: str
    options: str
    correct_answer: str
    question_row: pd.Series


class _RunWriter:
    def __init__(
        self,
        paths: RunPaths,
        results_state: dict[tuple[str, int], dict],
        n_trials: int,
    ) -> None:
        self.paths = paths
        self.results_state = results_state
        self.n_trials = n_trials
        self._lock = threading.Lock()
        self._done_keys: set[tuple[int, str, int]] = set()

    def seed_done_keys(self, keys: set[tuple[int, str, int]]) -> None:
        self._done_keys = set(keys)

    def is_done(self, trial: int, model: str, question_id: int) -> bool:
        return (trial, model, question_id) in self._done_keys

    def record(self, record: dict) -> None:
        with self._lock:
            trial = int(record["trial"])
            model = str(record["llm_model"])
            qid = int(record["question_id"])
            key = (trial, model, qid)
            if key in self._done_keys:
                return

            frame = pd.DataFrame([record], columns=RAW_RESPONSE_COLUMNS)
            write_header = not self.paths.raw_responses_csv.exists()
            frame.to_csv(
                self.paths.raw_responses_csv,
                mode="a",
                header=write_header,
                index=False,
            )

            result_key = _result_key(model, qid)
            self.results_state[result_key][f"it_{trial}_ans"] = record["parsed_answer"]
            self._done_keys.add(key)
            self._flush_results_unlocked()

    def _flush_results_unlocked(self) -> None:
        rows = [
            _finalize_result_row(dict(row), self.n_trials)
            for row in self.results_state.values()
        ]
        _write_results_csv(self.paths.results_csv, rows, self.n_trials)


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


def get_run_paths(run_id: str | None = None, *, must_exist: bool = False) -> RunPaths:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / stamp
    if must_exist and not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        results_csv=run_dir / "results.csv",
        raw_responses_csv=run_dir / "raw_responses.csv",
        metadata_json=run_dir / "run_metadata.json",
    )


def find_resumable_run() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    candidates = [
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and (path / "run_metadata.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


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


def _load_existing_raw_keys(raw_csv: Path) -> set[tuple[int, str, int]]:
    if not raw_csv.exists():
        return set()
    df = pd.read_csv(raw_csv)
    return {
        (int(row.trial), str(row.llm_model), int(row.question_id))
        for row in df.itertuples(index=False)
    }


def _sync_results_from_raw(
    raw_csv: Path,
    results_state: dict[tuple[str, int], dict],
) -> None:
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    for row in df.itertuples(index=False):
        key = _result_key(str(row.llm_model), int(row.question_id))
        if key not in results_state:
            continue
        trial_col = f"it_{int(row.trial)}_ans"
        parsed = "" if pd.isna(row.parsed_answer) else str(row.parsed_answer)
        results_state[key][trial_col] = parsed


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


def _expected_calls(models: list[str], question_count: int) -> int:
    return len(models) * question_count


def _trial_progress(
    trial: int,
    done_keys: set[tuple[int, str, int]],
    models: list[str],
    question_ids: list[int],
) -> tuple[int, int]:
    expected = _expected_calls(models, len(question_ids))
    done = sum(
        1
        for model in models
        for qid in question_ids
        if (trial, model, qid) in done_keys
    )
    return done, expected


def _completed_trials(
    trials: int,
    done_keys: set[tuple[int, str, int]],
    models: list[str],
    question_ids: list[int],
) -> list[int]:
    completed: list[int] = []
    for trial in range(1, trials + 1):
        done, expected = _trial_progress(trial, done_keys, models, question_ids)
        if done >= expected:
            completed.append(trial)
    return completed


def _load_metadata(paths: RunPaths) -> dict:
    if not paths.metadata_json.exists():
        return {}
    return json.loads(paths.metadata_json.read_text(encoding="utf-8"))


def _save_metadata(paths: RunPaths, metadata: dict) -> None:
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    paths.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _run_task(task: EvalTask) -> dict:
    prompt = format_prompt(task.question_row)
    output = chat_with_retry(prompt, model=task.model)
    pred = extract_answer_from_row(output, task.options) or ""
    return {
        "trial": task.trial,
        "llm_model": task.model,
        "question_id": task.question_id,
        "question_category": task.category,
        "correct_answer": task.correct_answer,
        "prompt": prompt,
        "raw_output": output,
        "parsed_answer": pred,
    }


def _build_pending_tasks(
    trial: int,
    models: list[str],
    df: pd.DataFrame,
    writer: _RunWriter,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for model in models:
        for _, row in df.iterrows():
            qid = int(row["question_id"])
            if writer.is_done(trial, model, qid):
                continue
            tasks.append(
                EvalTask(
                    trial=trial,
                    model=model,
                    question_id=qid,
                    category=row["category"],
                    options=row["options"],
                    correct_answer=str(row["answer"]).strip().upper(),
                    question_row=row,
                )
            )
    return tasks


def evaluate_run(
    models: list[str],
    *,
    dataset: pd.DataFrame | None = None,
    n_trials: int | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    continue_run: bool = False,
    max_workers: int | None = None,
) -> RunPaths:
    """Run trial-by-trial with parallel API calls and resumable progress."""
    cfg = load_inference_config()
    trials = n_trials if n_trials is not None else int(cfg["n_trials"])
    workers = max_workers if max_workers is not None else int(cfg.get("max_workers", 4))

    if continue_run:
        if run_id:
            paths = get_run_paths(run_id, must_exist=True)
        else:
            latest = find_resumable_run()
            if latest is None:
                raise FileNotFoundError(
                    "No resumable run found. Start a new run or pass --run-id."
                )
            paths = get_run_paths(latest.name, must_exist=True)
        metadata = _load_metadata(paths)
        if metadata.get("models"):
            models = list(metadata["models"])
        if metadata.get("n_trials"):
            trials = int(metadata["n_trials"])
        if metadata.get("max_workers") and max_workers is None:
            workers = int(metadata["max_workers"])
        print(f"Continuing run: {paths.run_dir}")
        df = load_dataset() if dataset is None else dataset
    else:
        paths = get_run_paths(run_id=run_id)
        metadata = {}
        df = dataset if dataset is not None else load_dataset()
        if limit is not None:
            df = df.head(limit)

    question_ids = [int(qid) for qid in df["question_id"].tolist()]
    done_keys = _load_existing_raw_keys(paths.raw_responses_csv)
    results_state = _load_results_state(paths.results_csv, models, df, trials)
    _sync_results_from_raw(paths.raw_responses_csv, results_state)

    writer = _RunWriter(paths, results_state, trials)
    writer.seed_done_keys(done_keys)

    completed = _completed_trials(trials, done_keys, models, question_ids)
    metadata.update(
        {
            "models": models,
            "n_trials": trials,
            "max_workers": workers,
            "inference_config": cfg,
            "dataset_rows": len(df),
            "completed_trials": completed,
            "status": "in_progress",
        }
    )
    if not metadata.get("started_at"):
        metadata["started_at"] = datetime.now(timezone.utc).isoformat()
    _save_metadata(paths, metadata)

    expected_per_trial = _expected_calls(models, len(df))
    total_models = len(models)
    total_questions = len(df)

    for trial in range(1, trials + 1):
        done, expected = _trial_progress(trial, writer._done_keys, models, question_ids)
        if done >= expected:
            print(f"\n=== Trial {trial}/{trials}: already complete, skipping ===")
            continue

        pending = _build_pending_tasks(trial, models, df, writer)
        print(
            f"\n=== Trial {trial}/{trials}: "
            f"{len(pending)} calls remaining ({done}/{expected} already done) ==="
        )
        metadata["current_trial"] = trial
        metadata["completed_trials"] = _completed_trials(
            trials, writer._done_keys, models, question_ids
        )
        _save_metadata(paths, metadata)

        completed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                record = future.result()
                writer.record(record)
                completed_count += 1
                print(
                    f"[trial {trial}/{trials}] [{task.model}] "
                    f"question id={task.question_id} "
                    f"({completed_count}/{len(pending)})"
                )

        done, expected = _trial_progress(trial, writer._done_keys, models, question_ids)
        if done >= expected:
            completed = _completed_trials(trials, writer._done_keys, models, question_ids)
            metadata["completed_trials"] = completed
            metadata["current_trial"] = trial
            metadata["status"] = "in_progress"
            print(f"Trial {trial} complete ({expected_per_trial} calls).")
        else:
            metadata["status"] = "interrupted"
            metadata["completed_trials"] = _completed_trials(
                trials, writer._done_keys, models, question_ids
            )
            _save_metadata(paths, metadata)
            print(
                f"Trial {trial} incomplete ({done}/{expected}). "
                f"Re-run with --continue to resume this trial only."
            )
            return paths

        _save_metadata(paths, metadata)

    metadata["status"] = "complete"
    metadata["current_trial"] = trials
    metadata["completed_trials"] = list(range(1, trials + 1))
    _save_metadata(paths, metadata)
    print(f"\nAll trials complete. Run directory: {paths.run_dir}")
    return paths
