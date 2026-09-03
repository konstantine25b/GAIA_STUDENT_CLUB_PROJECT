"""Load dataset and build MMLU-Pro prompts."""

import json
from pathlib import Path

import pandas as pd

from src.paths import DATASET_CSV

LETTERS = "ABCDEFGHIJ"

INSTRUCTION_TEMPLATE = (
    "The following are multiple choice questions about {category}. "
    "Reason through the question, then output ONLY a JSON object with this exact shape:\n"
    '{{"reasoning": "<your reasoning>", "answer": "<letter>"}}\n'
    'The "answer" value must be exactly one letter such as A or C, and nothing else.\n\n'
)


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or DATASET_CSV
    return pd.read_csv(path)


def format_question_block(question: str, options: list[str]) -> str:
    block = f"Question: {question}\nOptions:\n"
    for i, opt in enumerate(options):
        block += f"{LETTERS[i]}. {opt}\n"
    return block


def format_prompt(row: pd.Series) -> str:
    options = json.loads(row["options"])
    instruction = INSTRUCTION_TEMPLATE.format(category=row["category"])
    return instruction + format_question_block(row["question"], options)


def load_models(models_file: Path) -> list[str]:
    lines = models_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
