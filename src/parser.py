"""Answer extraction for MMLU-Pro model outputs."""

from __future__ import annotations

import json
import re
from typing import Sequence

LETTERS = "ABCDEFGHIJ"

CHOICE_PATTERN = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)
ANSWER_LINE_PATTERN = re.compile(r".*[aA]nswer:\s*([A-J])")
FINAL_LETTER_PATTERN = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)
BOXED_MARKER = r"\boxed{"
OPTION_REF_PATTERN = re.compile(
    r"(?:option|choice)\s*\(?([A-J])\)?",
    re.IGNORECASE,
)
LATEX_TEXT_PATTERN = re.compile(r"\\text\{([^}]*)\}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = LATEX_TEXT_PATTERN.sub(r"\1", text)
    text = text.replace("$", "")
    text = text.replace("\\,", "")
    text = text.replace("{", "").replace("}", "")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _letter_from_value(value: str, options: Sequence[str]) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None

    if len(cleaned) == 1 and cleaned.upper() in LETTERS[: len(options)]:
        return cleaned.upper()

    norm_value = _normalize_text(cleaned)
    if len(norm_value) == 1 and norm_value.upper() in LETTERS[: len(options)]:
        return norm_value.upper()

    for i, option in enumerate(options):
        if _normalize_text(option) == norm_value:
            return LETTERS[i]

    # Match when the model returns a numeric or short fragment of an option.
    for i, option in enumerate(options):
        norm_option = _normalize_text(option)
        if norm_value and (norm_value == norm_option or norm_value in norm_option):
            return LETTERS[i]

    return None


def _extract_boxed_values(text: str) -> list[str]:
    values: list[str] = []
    start = 0
    while True:
        idx = text.find(BOXED_MARKER, start)
        if idx == -1:
            break
        i = idx + len(BOXED_MARKER)
        depth = 1
        chars: list[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth > 0:
                chars.append(ch)
            i += 1
        values.append("".join(chars))
        start = i + 1
    return values


def _extract_from_boxed(text: str, options: Sequence[str] | None) -> str | None:
    matches = _extract_boxed_values(text)
    if not matches:
        return None

    for raw in reversed(matches):
        value = raw.strip()
        if options:
            letter = _letter_from_value(value, options)
            if letter:
                return letter
        if len(value) == 1 and value.upper() in LETTERS:
            return value.upper()
    return None


def _extract_from_option_refs(text: str) -> str | None:
    matches = OPTION_REF_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def extract_answer(text: str, options: Sequence[str] | None = None) -> str | None:
    """Parse model output into a single letter A-J."""
    if not text:
        return None

    cleaned = text.replace("**", "")
    tail = cleaned[-800:]

    if options:
        letter = _extract_from_boxed(tail, options)
        if letter:
            return letter

    match = CHOICE_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()

    match = ANSWER_LINE_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()

    letter = _extract_from_option_refs(tail)
    if letter:
        return letter

    if options:
        letter = _extract_from_boxed(cleaned, options)
        if letter:
            return letter

        # "which is 86", "is 64 seconds", "matches option F" style endings.
        for raw in re.findall(
            r"(?:is|equals|matches|corresponds to)\s+([^\n.]{1,80})",
            tail,
            flags=re.IGNORECASE,
        ):
            letter = _letter_from_value(raw.strip(), options)
            if letter:
                return letter

    match = FINAL_LETTER_PATTERN.search(cleaned)
    if match:
        letter = match.group(0).upper()
        if options is None or letter in LETTERS[: len(options)]:
            return letter

    return None


def extract_answer_from_row(text: str, options_json: str) -> str | None:
    """Parse answer using options stored in the dataset row."""
    options = json.loads(options_json)
    return extract_answer(text, options)
