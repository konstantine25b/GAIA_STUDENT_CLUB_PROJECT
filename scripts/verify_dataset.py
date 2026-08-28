"""Verify the initial 280-sample MMLU-Pro dataset."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import setup_project

setup_project()

from src.paths import DATASET_CSV

EXPECTED_ROWS = 280
EXPECTED_PER_CATEGORY = 20
EXPECTED_CATEGORIES = [
    "math",
    "physics",
    "chemistry",
    "law",
    "engineering",
    "other",
    "economics",
    "health",
    "psychology",
    "business",
    "biology",
    "philosophy",
    "computer science",
    "history",
]
REQUIRED_COLUMNS = [
    "question_id",
    "question",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
]
LETTERS = "ABCDEFGHIJ"


def verify_dataset(csv_path: Path | None = None) -> None:
    path = csv_path or DATASET_CSV
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    errors: list[str] = []

    if len(df) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {len(df)}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")

    if df["question_id"].duplicated().any():
        dupes = df[df["question_id"].duplicated(keep=False)]["question_id"].tolist()
        errors.append(f"duplicate question_id values: {dupes[:10]}")

    counts = df["category"].value_counts()
    for category in EXPECTED_CATEGORIES:
        count = int(counts.get(category, 0))
        if count != EXPECTED_PER_CATEGORY:
            errors.append(
                f"category '{category}': expected {EXPECTED_PER_CATEGORY}, got {count}"
            )

    extra_categories = set(df["category"].unique()) - set(EXPECTED_CATEGORIES)
    if extra_categories:
        errors.append(f"unexpected categories: {sorted(extra_categories)}")

    for idx, row in df.iterrows():
        try:
            options = json.loads(row["options"])
        except json.JSONDecodeError:
            errors.append(f"row {idx}: invalid options JSON")
            continue
        if not isinstance(options, list) or len(options) < 2:
            errors.append(
                f"question_id {row['question_id']}: expected >=2 options, got {len(options)}"
            )
            continue
        if len(options) > 10:
            errors.append(
                f"question_id {row['question_id']}: expected <=10 options, got {len(options)}"
            )
        answer = str(row["answer"]).strip().upper()
        valid_letters = LETTERS[: len(options)]
        if answer not in valid_letters:
            errors.append(
                f"question_id {row['question_id']}: answer '{answer}' "
                f"not in valid letters {valid_letters}"
            )
        elif int(row["answer_index"]) != valid_letters.index(answer):
            errors.append(
                f"question_id {row['question_id']}: answer_index does not match answer"
            )

    if errors:
        print(f"FAILED: {path}")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    print(f"OK: {path}")
    print(f"  rows: {len(df)}")
    print(f"  categories: {len(EXPECTED_CATEGORIES)} x {EXPECTED_PER_CATEGORY}")
    print(f"  unique question_ids: {df['question_id'].nunique()}")


if __name__ == "__main__":
    verify_dataset()
