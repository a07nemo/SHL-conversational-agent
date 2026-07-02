from __future__ import annotations

import json
import os
import re

import httpx

# Groq exposes an OpenAI-compatible /chat/completions endpoint. Swapping to
# another provider (OpenAI, together.ai, Fireworks, etc.) only requires
# changing BASE_URL / MODEL / the auth header below - the rest of the app
# talks to this module through `call_llm_json` only.
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("GROQ_API_KEY", "")

# Leave headroom under the platform's 30s per-call timeout (network + our
# own processing also needs a slice of that budget).
REQUEST_TIMEOUT_SECONDS = 20.0


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Defensively pull a JSON object out of a model response.

    Models occasionally wrap JSON in markdown fences or add stray prose
    despite instructions. We try strict parsing first, then fall back to
    locating the outermost {...} block.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Could not parse JSON from LLM response: {text[:500]!r}")


async def call_llm_json(system_prompt: str, messages: list[dict]) -> dict:
    """Call the LLM with a system prompt + chat history, expecting a JSON object back."""
    if not API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not set. Export it before running the service."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.post(GROQ_BASE_URL, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM HTTP error {e.response.status_code}: {e.response.text[:300]}")
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out")
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected LLM response shape: {data}")

    return _extract_json(content)
