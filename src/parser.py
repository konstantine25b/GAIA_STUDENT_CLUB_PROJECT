"""Extract the letter from a JSON model response."""

from __future__ import annotations

import json
import re
from typing import Sequence

LETTERS = "ABCDEFGHIJ"
FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _load_json_object(text: str) -> dict | None:
    cleaned = FENCE_PATTERN.sub("", text.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def extract_answer(text: str, options: Sequence[str] | None = None) -> str | None:
    """Read {"answer": "C"} from JSON. The value must be exactly one letter A-J."""
    if not text:
        return None

    obj = _load_json_object(text)
    if obj is None:
        return None

    raw = obj.get("answer")
    if raw is None:
        return None

    letter = str(raw).strip().upper()
    if len(letter) != 1 or letter not in LETTERS:
        return None
    if options is not None and letter not in LETTERS[: len(options)]:
        return None
    return letter


def extract_answer_from_row(text: str, options_json: str) -> str | None:
    """Parse answer using options stored in the dataset row."""
    options = json.loads(options_json)
    return extract_answer(text, options)
