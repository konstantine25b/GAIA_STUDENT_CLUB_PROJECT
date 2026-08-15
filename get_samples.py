"""Download MMLU-Pro and write 20 random samples per category to CSV."""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "TIGER-Lab/MMLU-Pro"
SPLIT = "test"
N_PER_CATEGORY = 20
SEED = 42
OUT_PATH = Path("data/mmlu_pro_sample_20_per_category.csv")
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
    OUT_PATH.parent.mkdir(exist_ok=True)
    sampled.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(sampled)} rows)")


if __name__ == "__main__":
    main()
