"""OpenAI-compatible client for the Gemini LiteLLM proxy."""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.paths import INFERENCE_CONFIG

load_dotenv()

BASE_URL = os.environ.get(
    "LITELLM_BASE_URL",
    "https://gemini-litellm-proxy-production.up.railway.app",
).rstrip("/")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")

SAMPLING_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
)

# Force a JSON object so the letter is read from the "answer" field, not regex.
JSON_ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mmlu_pro_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {
                    "type": "string",
                    "enum": list("ABCDEFGHIJ"),
                    "description": "Exactly one letter such as A or C, and nothing else.",
                },
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False,
        },
    },
}

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=f"{BASE_URL}/v1",
)


def load_inference_config() -> dict[str, Any]:
    with INFERENCE_CONFIG.open(encoding="utf-8") as f:
        return json.load(f)


def _sampling_kwargs(
    cfg: dict[str, Any],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    omit_max_tokens: bool = False,
) -> dict[str, Any]:
    """Build API sampling args. Omitted keys use provider defaults."""
    overrides = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    sampling = cfg.get("sampling") or {}
    params: dict[str, Any] = {}
    for key in SAMPLING_KEYS:
        value = overrides[key]
        if value is None:
            if key == "max_tokens" and omit_max_tokens:
                continue
            value = sampling.get(key)
        if value is not None:
            params[key] = value
    return params


def chat(
    prompt: str,
    model: str | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    omit_max_tokens: bool = False,
) -> str:
    cfg = load_inference_config()
    kwargs = _sampling_kwargs(
        cfg,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        omit_max_tokens=omit_max_tokens,
    )
    kwargs["response_format"] = JSON_ANSWER_RESPONSE_FORMAT
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content or ""


def chat_with_retry(
    prompt: str,
    model: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    omit_max_tokens: bool = False,
) -> str:
    cfg = load_inference_config()
    attempts = int(cfg.get("retry_on_api_error", 3))
    delay = float(cfg.get("retry_delay_seconds", 2))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return chat(
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                omit_max_tokens=omit_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - retry any API failure
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
    raise RuntimeError(f"API failed after {attempts} attempts for model={model}") from last_error
