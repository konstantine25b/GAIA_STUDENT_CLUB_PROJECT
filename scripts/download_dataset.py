"""Download MMLU-Pro and write 20 random samples per category to CSV."""

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import setup_project

setup_project()
from src.paths import DATASET_CSV

DATASET_ID = "TIGER-Lab/MMLU-Pro"
SPLIT = "test"
N_PER_CATEGORY = 20
SEED = 42
COLUMNS = [
    "question_id",
    "question",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
]
CATEGORY_ORDER = [
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


def get_samples() -> pd.DataFrame:
    df = load_dataset(DATASET_ID, split=SPLIT).to_pandas()[COLUMNS]
    samples = []
    for category in CATEGORY_ORDER:
        subset = df[df["category"] == category]
        if len(subset) < N_PER_CATEGORY:
            raise ValueError(f"{category}: only {len(subset)} rows")
        samples.append(subset.sample(n=N_PER_CATEGORY, random_state=SEED))
    sampled = pd.concat(samples, ignore_index=True)
    sampled["options"] = sampled["options"].apply(
        lambda opts: json.dumps(list(opts), ensure_ascii=False)
    )
    return sampled


def main() -> None:
    sampled = get_samples()
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(DATASET_CSV, index=False)
    print(f"wrote {DATASET_CSV} ({len(sampled)} rows)")


if __name__ == "__main__":
    main()
