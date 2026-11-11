"""Thin xAI (OpenAI-compatible) client wrapper."""

from __future__ import annotations

from openai import OpenAI

from lawtracker import config


def make_client() -> OpenAI:
    return OpenAI(
        api_key=config.secret("xai_api_key"),
        base_url=config.XAI_BASE_URL,
        timeout=config.HTTP_TIMEOUT_S,
        max_retries=3,  # SDK backs off on 429/5xx
    )


def cost_usd(model: str, usage: dict) -> float:
    """Cost from a usage dict. Reasoning tokens (if any) are billed as output; cached prompt tokens at the cached rate."""
    p = config.LLM_PRICES[model]
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    # Whether completion_tokens already includes reasoning is provider-specific (check with `lawtracker verify xai`).
    # Add it anyway: the cost guard must over-estimate, never under-estimate. Zero for non-reasoning models.
    out = completion + reasoning
    return ((prompt - cached) * p["input"] + cached * p["cached_input"] + out * p["output"]) / 1_000_000
