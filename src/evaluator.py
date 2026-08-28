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

DEFAULT_RETRY_PARSE_LADDER = [2048, 4096, 8192, 16256]

RAW_RESPONSE_COLUMNS = [
    "trial",
    "llm_model",
    "question_id",
    "question_category",
    "correct_answer",
    "attempt",
    "max_tokens",
    "parse_ok",
    "prompt",
    "raw_output",
    "parsed_answer",
]

PARSE_LOG_COLUMNS = [
    "trial",
    "llm_model",
    "question_id",
    "attempt",
    "max_tokens",
    "parse_ok",
    "parsed_answer",
    "correct_answer",
]


@dataclass
class RunPaths:
    run_dir: Path
    results_csv: Path
    raw_responses_csv: Path
    parse_log_csv: Path
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
    attempt: int = 1
    max_tokens: int | None = None


class _RunWriter:
    def __init__(
        self,
        paths: RunPaths,
        results_state: dict[tuple[str, int], dict],
        n_trials: int,
        df: pd.DataFrame,
    ) -> None:
        self.paths = paths
        self.results_state = results_state
        self.n_trials = n_trials
        self.df_by_qid = {
            int(row["question_id"]): row for _, row in df.iterrows()
        }
        self._lock = threading.Lock()

    def _ensure_result_row(self, model: str, question_id: int, prompt: str) -> None:
        key = _result_key(model, question_id)
        if key in self.results_state:
            if not self.results_state[key].get("prompt"):
                self.results_state[key]["prompt"] = prompt
            return
        row = self.df_by_qid[question_id]
        result = _empty_result_row(model, row, self.n_trials)
        result["prompt"] = prompt
        self.results_state[key] = result

    def record(self, record: dict) -> None:
        with self._lock:
            trial = int(record["trial"])
            model = str(record["llm_model"])
            qid = int(record["question_id"])
            prompt = str(record["prompt"])

            self._append_csv(self.paths.raw_responses_csv, record, RAW_RESPONSE_COLUMNS)
            self._append_csv(self.paths.parse_log_csv, _parse_log_row(record), PARSE_LOG_COLUMNS)

            self._ensure_result_row(model, qid, prompt)
            result_row = self.results_state[_result_key(model, qid)]
            result_row[f"it_{trial}_raw"] = record.get("raw_output") or ""
            parsed = record.get("parsed_answer") or ""
            if parsed:
                result_row[f"it_{trial}_ans"] = parsed
            self._flush_results_unlocked()

    def _append_csv(self, path: Path, record: dict, columns: list[str]) -> None:
        frame = pd.DataFrame([record], columns=columns)
        write_header = not path.exists()
        frame.to_csv(path, mode="a", header=write_header, index=False)

    def _flush_results_unlocked(self) -> None:
        rows = [
            _finalize_result_row(dict(row), self.n_trials)
            for row in self.results_state.values()
        ]
        _write_results_csv(self.paths.results_csv, rows, self.n_trials)


def _parse_log_row(record: dict) -> dict:
    parsed = record.get("parsed_answer") or ""
    return {
        "trial": record["trial"],
        "llm_model": record["llm_model"],
        "question_id": record["question_id"],
        "attempt": record["attempt"],
        "max_tokens": record["max_tokens"],
        "parse_ok": bool(parsed),
        "parsed_answer": parsed,
        "correct_answer": record["correct_answer"],
    }


def _trial_columns(n_trials: int) -> list[str]:
    columns: list[str] = []
    for i in range(1, n_trials + 1):
        columns.append(f"it_{i}_ans")
        columns.append(f"it_{i}_raw")
    return columns


def results_columns(n_trials: int) -> list[str]:
    return [
        "llm_model",
        "question_category",
        "question_id",
        "options",
        "correct_answer",
        "prompt",
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
        parse_log_csv=run_dir / "parse_log.csv",
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
        "prompt": "",
        "correct_answered_num": 0,
        "accuracy": 0.0,
    }
    for i in range(1, n_trials + 1):
        result[f"it_{i}_ans"] = ""
        result[f"it_{i}_raw"] = ""
    return result


def _finalize_result_row(row: dict, n_trials: int) -> dict:
    gold = row["correct_answer"]
    answers = [row.get(f"it_{i}_ans") or "" for i in range(1, n_trials + 1)]
    correct_count = sum(1 for ans in answers if ans == gold)
    row["correct_answered_num"] = correct_count
    row["accuracy"] = correct_count / n_trials
    return row


def _write_results_csv(results_csv: Path, rows: list[dict], n_trials: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=results_columns(n_trials))
    df = df.sort_values(["llm_model", "question_id"]).reset_index(drop=True)
    df.to_csv(results_csv, index=False)


def _load_raw_df(raw_csv: Path) -> pd.DataFrame:
    if not raw_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw_csv)
    if "attempt" not in df.columns:
        df["attempt"] = 1
    if "max_tokens" not in df.columns:
        df["max_tokens"] = ""
    if "parse_ok" not in df.columns:
        df["parse_ok"] = df["parsed_answer"].notna() & (
            df["parsed_answer"].astype(str).str.strip() != ""
        )
    return df


def _sync_results_from_raw(
    raw_df: pd.DataFrame,
    results_state: dict[tuple[str, int], dict],
    df: pd.DataFrame,
    n_trials: int,
) -> None:
    if raw_df.empty:
        return
    df_by_qid = {int(row["question_id"]): row for _, row in df.iterrows()}
    grouped = raw_df.sort_values("attempt").groupby(
        ["trial", "llm_model", "question_id"], sort=False
    )
    for (trial, model, qid), group in grouped:
        key = _result_key(str(model), int(qid))
        if key not in results_state:
            row = df_by_qid[int(qid)]
            results_state[key] = _empty_result_row(str(model), row, n_trials)
        prompt = group.iloc[-1].get("prompt", "")
        if isinstance(prompt, str) and prompt and not results_state[key].get("prompt"):
            results_state[key]["prompt"] = prompt
        for _, attempt_row in group.iterrows():
            raw_output = (
                "" if pd.isna(attempt_row.raw_output) else str(attempt_row.raw_output)
            )
            if raw_output:
                results_state[key][f"it_{int(trial)}_raw"] = raw_output
            parsed = "" if pd.isna(attempt_row.parsed_answer) else str(attempt_row.parsed_answer)
            if parsed:
                results_state[key][f"it_{int(trial)}_ans"] = parsed


def _load_results_state(results_csv: Path, n_trials: int) -> dict[tuple[str, int], dict]:
    if not results_csv.exists():
        return {}
    state: dict[tuple[str, int], dict] = {}
    for record in pd.read_csv(results_csv).to_dict(orient="records"):
        key = _result_key(str(record["llm_model"]), int(record["question_id"]))
        for i in range(1, n_trials + 1):
            record.setdefault(f"it_{i}_ans", "")
            record.setdefault(f"it_{i}_raw", "")
        state[key] = record
    return state


def _retry_parse_ladder(cfg: dict) -> list[int]:
    value = cfg.get("retry_parse_max_tokens", DEFAULT_RETRY_PARSE_LADDER)
    if value is None:
        return list(DEFAULT_RETRY_PARSE_LADDER)
    if isinstance(value, int):
        return [value] if value > 0 else []
    return [int(item) for item in value if int(item) > 0]


def _is_parsed(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def _max_tokens_tried(rows: pd.DataFrame) -> list[int]:
    tried: list[int] = []
    if rows.empty or "max_tokens" not in rows.columns:
        return tried
    for value in rows["max_tokens"]:
        if value is None or value == "" or pd.isna(value):
            continue
        tried.append(int(value))
    return tried


def _next_retry_max_tokens(tried: list[int], ladder: list[int]) -> int | None:
    highest = max(tried) if tried else 0
    for tokens in ladder:
        if tokens > highest:
            return tokens
    return None


def _model_trial_progress(
    model: str,
    trial: int,
    raw_df: pd.DataFrame,
    question_ids: list[int],
    retry_ladder: list[int],
) -> tuple[int, int]:
    expected = len(question_ids)
    if raw_df.empty:
        return 0, expected
    done = 0
    for qid in question_ids:
        if _call_cycle_complete(raw_df, trial, model, qid, retry_ladder):
            done += 1
    return done, expected


def _call_cycle_complete(
    raw_df: pd.DataFrame,
    trial: int,
    model: str,
    question_id: int,
    retry_ladder: list[int],
) -> bool:
    rows = raw_df[
        (raw_df["trial"] == trial)
        & (raw_df["llm_model"] == model)
        & (raw_df["question_id"] == question_id)
    ]
    if rows.empty:
        return False
    if rows[rows["attempt"] == 1].empty:
        return False
    if any(_is_parsed(value) for value in rows["parsed_answer"]):
        return True
    return _next_retry_max_tokens(_max_tokens_tried(rows), retry_ladder) is None


def _trial_initial_complete(
    models: list[str],
    trial: int,
    raw_df: pd.DataFrame,
    question_ids: list[int],
) -> bool:
    if raw_df.empty:
        return False
    initial = raw_df[(raw_df["trial"] == trial) & (raw_df["attempt"] == 1)]
    have = {
        (str(model), int(qid))
        for model, qid in zip(initial["llm_model"], initial["question_id"])
    }
    expected = {(model, qid) for model in models for qid in question_ids}
    return expected.issubset(have)


def _trial_is_complete(
    models: list[str],
    trial: int,
    raw_df: pd.DataFrame,
    question_ids: list[int],
    retry_ladder: list[int],
) -> bool:
    for model in models:
        done, expected = _model_trial_progress(
            model, trial, raw_df, question_ids, retry_ladder
        )
        if done < expected:
            return False
    return True


def _completed_trials(
    trials: int,
    models: list[str],
    raw_df: pd.DataFrame,
    question_ids: list[int],
    retry_ladder: list[int],
) -> list[int]:
    return [
        trial
        for trial in range(1, trials + 1)
        if _trial_is_complete(models, trial, raw_df, question_ids, retry_ladder)
    ]


def _model_is_complete(
    model: str,
    trials: int,
    raw_df: pd.DataFrame,
    question_ids: list[int],
    retry_ladder: list[int],
) -> bool:
    for trial in range(1, trials + 1):
        done, expected = _model_trial_progress(
            model, trial, raw_df, question_ids, retry_ladder
        )
        if done < expected:
            return False
    return True


def _completed_models(
    models: list[str],
    trials: int,
    raw_df: pd.DataFrame,
    question_ids: list[int],
    retry_ladder: list[int],
) -> list[str]:
    return [
        model
        for model in models
        if _model_is_complete(model, trials, raw_df, question_ids, retry_ladder)
    ]


def _load_metadata(paths: RunPaths) -> dict:
    if not paths.metadata_json.exists():
        return {}
    return json.loads(paths.metadata_json.read_text(encoding="utf-8"))


def _save_metadata(paths: RunPaths, metadata: dict) -> None:
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    paths.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _default_max_tokens(cfg: dict) -> int | None:
    sampling = cfg.get("sampling") or {}
    value = sampling.get("max_tokens")
    return int(value) if value is not None else None


def _run_task(task: EvalTask) -> dict:
    prompt = format_prompt(task.question_row)
    output = chat_with_retry(prompt, model=task.model, max_tokens=task.max_tokens)
    pred = extract_answer_from_row(output, task.options) or ""
    return {
        "trial": task.trial,
        "llm_model": task.model,
        "question_id": task.question_id,
        "question_category": task.category,
        "correct_answer": task.correct_answer,
        "attempt": task.attempt,
        "max_tokens": task.max_tokens if task.max_tokens is not None else "",
        "parse_ok": bool(pred),
        "prompt": prompt,
        "raw_output": output,
        "parsed_answer": pred,
    }


def _build_initial_tasks(
    model: str,
    trial: int,
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    default_max_tokens: int | None,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        if not raw_df.empty:
            has_initial = not raw_df[
                (raw_df["trial"] == trial)
                & (raw_df["llm_model"] == model)
                & (raw_df["question_id"] == qid)
                & (raw_df["attempt"] == 1)
            ].empty
            if has_initial:
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
                attempt=1,
                max_tokens=default_max_tokens,
            )
        )
    return tasks


def _build_retry_tasks(
    models: list[str],
    trial: int,
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    retry_max_tokens: int,
    retry_ladder: list[int],
) -> list[EvalTask]:
    if retry_max_tokens <= 0 or raw_df.empty:
        return []
    df_by_qid = {int(row["question_id"]): row for _, row in df.iterrows()}
    tasks: list[EvalTask] = []
    for model in models:
        for qid, row in df_by_qid.items():
            rows = raw_df[
                (raw_df["trial"] == trial)
                & (raw_df["llm_model"] == model)
                & (raw_df["question_id"] == qid)
            ]
            if rows.empty:
                continue
            if any(_is_parsed(value) for value in rows["parsed_answer"]):
                continue
            if _next_retry_max_tokens(_max_tokens_tried(rows), retry_ladder) != retry_max_tokens:
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
                    attempt=int(rows["attempt"].max()) + 1,
                    max_tokens=retry_max_tokens,
                )
            )
    return tasks


def _run_pending_batch(
    pending: list[EvalTask],
    writer: _RunWriter,
    workers: int,
    *,
    trial: int,
    trials: int,
    label: str,
) -> bool:
    if not pending:
        return True

    completed_count = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(_run_task, task): task for task in pending}
    try:
        for future in as_completed(futures):
            task = futures[future]
            record = future.result()
            writer.record(record)
            completed_count += 1
            print(
                f"[{task.model}] trial {trial}/{trials} {label} "
                f"question id={task.question_id} "
                f"({completed_count}/{len(pending)})"
            )
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Resume with: --continue --run-id <id>")
        executor.shutdown(wait=False, cancel_futures=True)
        return False
    else:
        executor.shutdown(wait=True)
        return True


def _reload_raw(paths: RunPaths) -> pd.DataFrame:
    return _load_raw_df(paths.raw_responses_csv)


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
    """For each trial, run every model, then retry parse failures with rising max_tokens."""
    cfg = load_inference_config()
    trials = n_trials if n_trials is not None else int(cfg["n_trials"])
    workers = max_workers if max_workers is not None else int(cfg.get("max_workers", 4))
    default_max_tokens = _default_max_tokens(cfg)
    retry_ladder = _retry_parse_ladder(cfg)

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
    raw_df = _reload_raw(paths)
    results_state = _load_results_state(paths.results_csv, trials)
    _sync_results_from_raw(raw_df, results_state, df, trials)
    writer = _RunWriter(paths, results_state, trials, df)

    def _mark_interrupted() -> None:
        raw_now = _reload_raw(paths)
        metadata["status"] = "interrupted"
        metadata["completed_trials"] = _completed_trials(
            trials, models, raw_now, question_ids, retry_ladder
        )
        metadata["completed_models"] = _completed_models(
            models, trials, raw_now, question_ids, retry_ladder
        )
        _save_metadata(paths, metadata)

    metadata.update(
        {
            "models": models,
            "n_trials": trials,
            "max_workers": workers,
            "inference_config": cfg,
            "dataset_rows": len(df),
            "run_order": "trial_first",
            "retry_parse_max_tokens": retry_ladder,
            "completed_trials": _completed_trials(
                trials, models, raw_df, question_ids, retry_ladder
            ),
            "completed_models": _completed_models(
                models, trials, raw_df, question_ids, retry_ladder
            ),
            "status": "in_progress",
        }
    )
    if not metadata.get("started_at"):
        metadata["started_at"] = datetime.now(timezone.utc).isoformat()
    _save_metadata(paths, metadata)

    try:
        for trial in range(1, trials + 1):
            raw_df = _reload_raw(paths)
            if _trial_is_complete(models, trial, raw_df, question_ids, retry_ladder):
                print(f"\n=== Trial {trial}/{trials}: already complete, skipping ===")
                continue

            print(f"\n=== Trial {trial}/{trials} ===")
            metadata["current_trial"] = trial
            metadata["current_phase"] = "initial"
            _save_metadata(paths, metadata)

            for model_idx, model in enumerate(models, start=1):
                raw_df = _reload_raw(paths)
                initial = _build_initial_tasks(
                    model, trial, df, raw_df, default_max_tokens
                )
                if not initial:
                    print(
                        f"  Model {model_idx}/{len(models)} {model}: "
                        "initial already done"
                    )
                    continue
                print(
                    f"  Model {model_idx}/{len(models)} {model}: initial pass, "
                    f"{len(initial)} calls (max_tokens={default_max_tokens})"
                )
                metadata["current_model"] = model
                _save_metadata(paths, metadata)
                ok = _run_pending_batch(
                    initial,
                    writer,
                    workers,
                    trial=trial,
                    trials=trials,
                    label="initial",
                )
                if not ok:
                    _mark_interrupted()
                    return paths

            raw_df = _reload_raw(paths)
            if not _trial_initial_complete(models, trial, raw_df, question_ids):
                _mark_interrupted()
                print(
                    f"  Trial {trial}/{trials} initial pass incomplete. "
                    "Re-run with --continue."
                )
                return paths

            metadata["current_phase"] = "parse_retry"
            _save_metadata(paths, metadata)

            for retry_tokens in retry_ladder:
                raw_df = _reload_raw(paths)
                retries = _build_retry_tasks(
                    models, trial, df, raw_df, retry_tokens, retry_ladder
                )
                if not retries:
                    continue
                print(
                    f"  Trial {trial}/{trials}: parse retry, "
                    f"{len(retries)} calls (max_tokens={retry_tokens})"
                )
                metadata["current_retry_max_tokens"] = retry_tokens
                _save_metadata(paths, metadata)
                ok = _run_pending_batch(
                    retries,
                    writer,
                    workers,
                    trial=trial,
                    trials=trials,
                    label=f"parse-retry@{retry_tokens}",
                )
                if not ok:
                    _mark_interrupted()
                    return paths

            raw_df = _reload_raw(paths)
            if not _trial_is_complete(models, trial, raw_df, question_ids, retry_ladder):
                _mark_interrupted()
                print(
                    f"  Trial {trial}/{trials} incomplete. Re-run with --continue."
                )
                return paths

            metadata["completed_trials"] = _completed_trials(
                trials, models, raw_df, question_ids, retry_ladder
            )
            metadata["completed_models"] = _completed_models(
                models, trials, raw_df, question_ids, retry_ladder
            )
            _save_metadata(paths, metadata)
            print(
                f"  Trial {trial}/{trials} complete for all {len(models)} models."
            )

    except KeyboardInterrupt:
        _mark_interrupted()
        print("\nInterrupted — progress saved. Resume with: --continue --run-id <id>")
        return paths

    metadata["status"] = "complete"
    metadata["completed_trials"] = list(range(1, trials + 1))
    metadata["completed_models"] = list(models)
    _save_metadata(paths, metadata)
    print(f"\nAll trials complete. Run directory: {paths.run_dir}")
    return paths
