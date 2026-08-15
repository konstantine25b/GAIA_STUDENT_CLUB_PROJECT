"""Minimal OpenAI-compatible client for the Gemini LiteLLM proxy."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.environ.get(
    "LITELLM_BASE_URL",
    "https://gemini-litellm-proxy-production.up.railway.app",
).rstrip("/")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=f"{BASE_URL}/v1",
)


def chat(prompt: str, model: str | None = None, max_tokens: int = 256) -> str:
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(chat("Reply with one word: ok", max_tokens=8))
