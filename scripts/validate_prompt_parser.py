"""Pre-flight check for MMLU-Pro prompt format and answer parsing."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import setup_project

setup_project()

import pandas as pd

from src.dataset import format_prompt, load_dataset
from src.llm_client import chat_with_retry, load_inference_config
from src.parser import extract_answer_from_row


def default_validation_ids() -> list[int]:
    return [
        10167,
        5622,
        10532,
        10436,
        6501,
        7763,
        6604,
        3445,
        6813,
        6588,
        10850,
        11740,
        11011,
        3463,
        8058,
        7952,
        4433,
        6437,
        10783,
        3139,
    ]


def run_validation(model: str, question_ids: list[int]) -> pd.DataFrame:
    df = load_dataset()
    subset = df[df["question_id"].isin(question_ids)].copy()
    missing = set(question_ids) - set(subset["question_id"].tolist())
    if missing:
        raise ValueError(f"question_ids not in dataset: {sorted(missing)}")

    rows = []
    for _, row in subset.iterrows():
        prompt = format_prompt(row)
        output = chat_with_retry(prompt, model=model)
        pred = extract_answer_from_row(output, row["options"])
        gold = str(row["answer"]).strip().upper()
        rows.append(
            {
                "question_id": int(row["question_id"]),
                "category": row["category"],
                "gold": gold,
                "parsed": pred or "",
                "parse_ok": pred is not None,
                "correct": pred == gold if pred else False,
                "output_tail": output[-300:].replace("\n", "\\n"),
            }
        )
        print(
            f"id={row['question_id']} gold={gold} parsed={pred} "
            f"parse_ok={pred is not None} correct={pred == gold if pred else False}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prompt and parser.")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Model to test (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/validation"),
        help="Directory for validation report",
    )
    args = parser.parse_args()

    cfg = load_inference_config()
    print("Inference config:", json.dumps(cfg, indent=2))

    report = run_validation(args.model, default_validation_ids())
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"validation_{args.model.replace('/', '_')}.csv"
    report.to_csv(out_path, index=False)

    n = len(report)
    parse_rate = report["parse_ok"].mean()
    acc = report["correct"].mean()
    print(f"\nWrote {out_path}")
    print(f"Parse success: {report['parse_ok'].sum()}/{n} ({parse_rate:.1%})")
    print(f"Accuracy (1 trial): {report['correct'].sum()}/{n} ({acc:.1%})")


if __name__ == "__main__":
    main()
