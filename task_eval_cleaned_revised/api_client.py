import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 300
_BACKOFF_BASE = 5


def _post_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    *,
    max_retries: int = 5,
) -> dict:
    """429 n back-off handling"""
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning(
                "Network error (attempt %d/%d): %s", attempt + 1, max_retries, exc
            )
            time.sleep(_BACKOFF_BASE * (attempt + 1))
            continue

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", _BACKOFF_BASE * (attempt + 1)))
            logger.warning(
                "Rate limited (429). Sleeping %ds (attempt %d/%d)...",
                wait,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait)
            continue

        if not resp.ok:
            logger.error("API error %d: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Exceeded {max_retries} retries.")


def _extract_text(data: dict) -> str:
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content.strip()

    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response structure: {data}") from exc


def generate_groq(
    query: str,
    max_new_tokens: int,
    *,
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """Groq api implementation"""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    data = _post_with_retry(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_new_tokens,
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        max_retries=max_retries,
    )
    return _extract_text(data)


def generate_openrouter(
    query: str,
    max_new_tokens: int,
    *,
    model: str = "thinkingmachines/inkling:free",
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """OpenRouter api implementation"""

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    data = _post_with_retry(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_new_tokens,
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        max_retries=max_retries,
    )

    return _extract_text(data)


def generate_nvidia(
    query: str,
    max_new_tokens: int,
    *,
    model: str = "meta/llama-3.1-8b-instruct",
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """NVIDIA NIM api implementation."""
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    base_url = "https://nim.api.nvidia.com/v1"

    data = _post_with_retry(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_new_tokens,
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        max_retries=max_retries,
    )
    return _extract_text(data)


def resolve_generator(provider: str):
    mapping = {
        "groq": generate_groq,
        "openrouter": generate_openrouter,
        "nvidia": generate_nvidia,
    }
    try:
        return mapping[provider.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {', '.join(mapping)}"
        ) from exc
