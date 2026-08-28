"""Load dataset and build MMLU-Pro prompts."""

import json
from pathlib import Path

import pandas as pd

from src.paths import DATASET_CSV

LETTERS = "ABCDEFGHIJ"

# Concise prompt: shorter model outputs = lower API cost (cap with max_tokens in config).
INSTRUCTION_TEMPLATE = (
    'The following are multiple choice questions about {category}. '
    'Think briefly (at most a few sentences), then end with exactly: '
    '"The answer is (X)" where X is the correct letter A-J.\n\n'
)


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or DATASET_CSV
    return pd.read_csv(path)


def format_question_block(question: str, options: list[str]) -> str:
    block = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options):
        block += f"{LETTERS[i]}. {opt}\n"
    block += "Answer: "
    return block


def format_prompt(row: pd.Series) -> str:
    options = json.loads(row["options"])
    instruction = INSTRUCTION_TEMPLATE.format(category=row["category"])
    return instruction + format_question_block(row["question"], options)


def load_models(models_file: Path) -> list[str]:
    lines = models_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
