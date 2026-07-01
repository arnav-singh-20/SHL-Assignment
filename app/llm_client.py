"""Gemini REST API client (no SDK dependency, just requests)."""

import json
import logging
import os
import re
import time

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

logger = logging.getLogger("shl_agent.llm")

MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.6
REQUEST_TIMEOUT_SECONDS = 12


class LLMError(RuntimeError):
    pass


def call_llm(system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set")

    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            resp = requests.post(
                url,
                headers={"content-type": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as e:
                    raise LLMError(f"Unexpected Gemini response shape: {data}") from e
                logger.info("llm_call ok attempt=%d latency_ms=%d", attempt, elapsed_ms)
                return text.strip()

            retryable = resp.status_code == 429 or resp.status_code >= 500
            logger.warning(
                "llm_call failed attempt=%d status=%d retryable=%s latency_ms=%d",
                attempt, resp.status_code, retryable, elapsed_ms,
            )
            last_err = LLMError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
            if not retryable:
                raise last_err

        except requests.RequestException as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning("llm_call exception attempt=%d latency_ms=%d err=%s", attempt, elapsed_ms, e)
            last_err = LLMError(str(e))

        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)

    raise last_err or LLMError("LLM call failed for an unknown reason")


def call_llm_json(system: str, user: str, max_tokens: int = 700) -> dict:
    """Calls the LLM and parses a JSON object out of its reply, stripping any
    stray markdown fences or preamble the model might add."""
    raw = call_llm(system, user, max_tokens=max_tokens, temperature=0.0)
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise LLMError(f"No JSON object found in LLM output: {raw[:300]}")
    return json.loads(match.group(0))
