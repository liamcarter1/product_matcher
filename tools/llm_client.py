"""
ProductMatchPro - Dual-Provider LLM Client
Shared client for Anthropic (Claude) and OpenAI with provider-agnostic tier system.
Switch providers via LLM_PROVIDER env var ("anthropic" or "openai").
"""

import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Provider configuration ───────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Provider-agnostic tier constants
TIER_HIGH = "high"   # Critical extraction (ordering codes, vision)
TIER_MID = "mid"     # Medium tasks (specs, spool analysis, chat responses)
TIER_LOW = "low"     # Fast/cheap (query parsing, classification)

# Anthropic model mapping
_ANTHROPIC_MODELS = {
    TIER_HIGH: "claude-opus-4-20250514",
    TIER_MID: "claude-sonnet-4-20250514",
    TIER_LOW: "claude-haiku-4-20250514",
}

# OpenAI model mapping (fallback)
_OPENAI_MODELS = {
    TIER_HIGH: "gpt-4.1",
    TIER_MID: "gpt-4.1-mini",
    TIER_LOW: "gpt-4o-mini",
}

# Vision model mapping (for image-heavy calls)
_ANTHROPIC_VISION = {
    TIER_HIGH: "claude-opus-4-20250514",
    TIER_MID: "claude-sonnet-4-20250514",
    TIER_LOW: "claude-haiku-4-20250514",
}
_OPENAI_VISION = {
    TIER_HIGH: "gpt-4o",
    TIER_MID: "gpt-4o",
    TIER_LOW: "gpt-4o-mini",
}

# ── Singleton clients ────────────────────────────────────────────────────

_anthropic_client = None
_openai_client = None


def get_provider() -> str:
    """Return the currently active provider name."""
    return LLM_PROVIDER


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def get_client():
    """Return the active provider's client (for advanced use / testing)."""
    if LLM_PROVIDER == "anthropic":
        return _get_anthropic()
    return _get_openai()


def _resolve_model(tier: str, vision: bool = False) -> str:
    """Map a tier to the correct model name for the active provider."""
    if LLM_PROVIDER == "anthropic":
        table = _ANTHROPIC_VISION if vision else _ANTHROPIC_MODELS
    else:
        table = _OPENAI_VISION if vision else _OPENAI_MODELS
    return table.get(tier, table[TIER_MID])


# ── Core LLM call functions ─────────────────────────────────────────────

def call_llm(
    tier: str,
    system: str,
    user_content,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    retries: int = 3,
    backoff_base: float = 2.0,
    vision: bool = False,
) -> str:
    """Make an LLM call with retry logic. Returns text response.

    Args:
        tier: TIER_HIGH, TIER_MID, or TIER_LOW
        system: System prompt string
        user_content: str or list of content blocks (for vision)
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
        retries: Number of retry attempts
        backoff_base: Exponential backoff base in seconds
        vision: Whether this is a vision call (affects model selection)
    """
    model = _resolve_model(tier, vision=vision)

    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(model, system, user_content,
                               max_tokens=max_tokens, temperature=temperature,
                               retries=retries, backoff_base=backoff_base)
    else:
        return _call_openai(model, system, user_content,
                            max_tokens=max_tokens, temperature=temperature,
                            retries=retries, backoff_base=backoff_base)


def call_llm_json(
    tier: str,
    system: str,
    user_content,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    vision: bool = False,
) -> dict | list:
    """Call LLM and parse response as JSON.

    For Anthropic: appends JSON instruction to system prompt, strips markdown
    fences, retries with fix-up prompt on parse failure.
    For OpenAI: uses response_format={"type": "json_object"}.
    """
    model = _resolve_model(tier, vision=vision)

    if LLM_PROVIDER == "openai":
        return _call_openai_json(model, system, user_content,
                                 max_tokens=max_tokens, temperature=temperature)

    # Anthropic path: no native JSON mode
    enhanced_system = system + "\n\nReturn ONLY valid JSON. No explanation, no markdown code fences."
    raw = _call_anthropic(model, enhanced_system, user_content,
                          max_tokens=max_tokens, temperature=temperature)
    return _parse_json_response(raw, model, max_tokens)


def call_llm_tool(
    tier: str,
    system: str,
    user_content,
    tool_schema: dict,
    *,
    max_tokens: int = 8192,
    temperature: float = 0.1,
    vision: bool = False,
) -> dict:
    """Call LLM with tool use for guaranteed structured output.

    For Anthropic: uses tool_choice to force tool invocation.
    For OpenAI: falls back to call_llm_json.
    """
    if LLM_PROVIDER == "openai":
        # OpenAI fallback: just use JSON mode with the system prompt
        return call_llm_json(tier, system, user_content,
                             max_tokens=max_tokens, temperature=temperature,
                             vision=vision)

    model = _resolve_model(tier, vision=vision)
    client = _get_anthropic()

    # Normalise user_content
    if isinstance(user_content, str):
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [{"role": "user", "content": user_content}]

    try:
        response = client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("No tool_use block in response")
    except Exception as e:
        logger.warning("Tool use failed (%s), falling back to JSON mode: %s", model, e)
        return call_llm_json(tier, system, user_content,
                             max_tokens=max_tokens, temperature=temperature,
                             vision=vision)


# ── Anthropic internals ──────────────────────────────────────────────────

def _call_anthropic(
    model: str,
    system: str,
    user_content,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Raw Anthropic API call with retry."""
    import anthropic

    client = _get_anthropic()
    if isinstance(user_content, str):
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [{"role": "user", "content": user_content}]

    last_error = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            wait = backoff_base ** attempt
            logger.warning("Anthropic rate-limited, waiting %.1fs (attempt %d/%d)",
                           wait, attempt + 1, retries)
            time.sleep(wait)
            last_error = e
        except anthropic.APIError as e:
            if attempt == retries - 1:
                raise
            logger.warning("Anthropic API error (attempt %d/%d): %s",
                           attempt + 1, retries, e)
            time.sleep(backoff_base ** attempt)
            last_error = e

    raise RuntimeError(f"Anthropic call failed after {retries} retries: {last_error}")


# ── OpenAI internals ─────────────────────────────────────────────────────

def _call_openai(
    model: str,
    system: str,
    user_content,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Raw OpenAI API call with retry."""
    client = _get_openai()

    if isinstance(user_content, str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff_base ** attempt
            logger.warning("OpenAI API error (attempt %d/%d): %s",
                           attempt + 1, retries, e)
            time.sleep(wait)
            last_error = e

    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_error}")


def _call_openai_json(
    model: str,
    system: str,
    user_content,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> dict | list:
    """OpenAI call with JSON mode."""
    client = _get_openai()

    if isinstance(user_content, str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content)


# ── JSON parsing helpers ─────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove first line (```json) and last line (```)
        filtered = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            filtered.append(line)
        stripped = "\n".join(filtered)
    return stripped.strip()


def _parse_json_response(raw: str, model: str, max_tokens: int) -> dict | list:
    """Parse a raw LLM response as JSON, with fix-up retry on failure."""
    text = _strip_markdown_fences(raw)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed from %s, attempting fix-up", model)
        fix_prompt = (
            "The following text should be valid JSON but has syntax errors. "
            "Fix it and return ONLY valid JSON, nothing else:\n\n"
            + raw[:8000]
        )
        fixed_raw = _call_anthropic(
            model,
            "Return ONLY valid JSON. No explanation.",
            fix_prompt,
            max_tokens=max_tokens,
        )
        fixed = _strip_markdown_fences(fixed_raw)
        return json.loads(fixed)
