"""Provider adapters behind one small interface (docs/llm-providers-design.md section 1).

Adapters take LLMRequests and return LLMResults. They never touch the database or S3, never log request or
response bodies, and read only their own provider's key, lazily. Every answer is validated against the pydantic
schema here, whatever the provider promised: Anthropic's SDK drops constraints such as max_length from the wire
schema, so the pydantic check is the contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from functools import cached_property

from pydantic import ValidationError

from lawtracker import config
from lawtracker.llm import prices
from lawtracker.llm.types import CUSTOM_ID, BatchInfo, LLMRequest, LLMResult, Usage

TIMEOUT_S = 180.0
# JSON answer plus thinking. Sonnet 5.5 used at most 915 on the 2026-11-21 test; a cut answer is reported as
# `truncated`, never parsed, and this cap bounds what one runaway answer can cost.
MAX_OUTPUT_TOKENS = 4000


def _parse(schema, text: str):
    try:
        return schema.model_validate_json(text or ""), None
    except ValidationError:
        return None, "invalid"


class OpenAICompatProvider:
    """xAI and OpenAI: the same chat.completions request with a strict JSON schema; both cache a repeated prompt
    prefix automatically. Live calls only (no batch adapter yet; xAI's batch discount is unverified)."""

    supports_batch = False

    def __init__(self, name: str, key: str, base_url: str | None, thinking_in_completion: bool):
        self.name, self.key, self.base_url = name, key, base_url
        self.thinking_in_completion = thinking_in_completion  # OpenAI counts reasoning inside completion_tokens

    @cached_property
    def client(self):
        from openai import OpenAI

        return OpenAI(api_key=config.secret(self.key), base_url=self.base_url, timeout=TIMEOUT_S, max_retries=2)

    def run_one(self, req: LLMRequest, *, model: str) -> LLMResult:
        from openai.lib._pydantic import to_strict_json_schema

        kw = {"temperature": 0} if "non-reasoning" in model else {}  # reasoning models take no sampling settings
        try:
            r = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "classification", "schema": to_strict_json_schema(req.schema), "strict": True}},
                **kw,
            )  # fmt: skip
        except Exception as e:  # noqa: BLE001 - reported as a status; the caller retries or leaves the bill pending
            return LLMResult(req.custom_id, "error", error=type(e).__name__)
        u = r.usage.model_dump() if r.usage else {}
        thinking = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        out = (u.get("completion_tokens") or 0) + (0 if self.thinking_in_completion else thinking)
        usage = Usage(input=u.get("prompt_tokens") or 0,
                      cached_read=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      output=out, thinking=thinking)  # fmt: skip
        res = LLMResult(req.custom_id, "ok", usage=usage, returned_model=r.model,
                        cost_usd=prices.cost(self.name, model, usage))  # fmt: skip
        choice = r.choices[0]
        if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
            res.status = "refusal"
        elif choice.finish_reason == "length":
            res.status = "truncated"
        else:
            res.parsed, err = _parse(req.schema, choice.message.content)
            if err:
                res.status = err
        return res

    def submit_batch(self, reqs: Sequence[LLMRequest], *, model: str):
        raise NotImplementedError(f"{self.name}: no batch adapter")


class AnthropicProvider:
    """Claude via the Messages API: the system prompt is one cached block (cache_control), the answer is a JSON
    schema output; batches through the Message Batches API (50% off, results within 24 hours)."""

    name = "anthropic"
    supports_batch = True

    @cached_property
    def client(self):
        import anthropic

        return anthropic.Anthropic(api_key=config.secret("anthropic_api_key"), timeout=TIMEOUT_S, max_retries=2)

    @staticmethod
    def params(req: LLMRequest, model: str) -> dict:
        import anthropic

        return {
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": req.user}],
            "output_config": {"format": {"type": "json_schema", "schema": anthropic.transform_schema(req.schema)}},
        }

    def result(self, cid: str, msg, schema, model: str, *, batch: bool) -> LLMResult:
        u = msg.usage
        cr, cw = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        usage = Usage(input=u.input_tokens + cr + cw, cached_read=cr, cache_write=cw, output=u.output_tokens)
        res = LLMResult(cid, "ok", usage=usage, returned_model=msg.model,
                        cost_usd=prices.cost(self.name, model, usage, batch=batch))  # fmt: skip
        if msg.stop_reason == "refusal":
            res.status = "refusal"
        elif msg.stop_reason == "max_tokens":
            res.status = "truncated"
        else:
            text = next((b.text for b in msg.content if b.type == "text"), "")
            res.parsed, err = _parse(schema, text)
            if err:
                res.status = err
        return res

    def run_one(self, req: LLMRequest, *, model: str) -> LLMResult:
        try:
            msg = self.client.messages.create(**self.params(req, model))
        except Exception as e:  # noqa: BLE001
            return LLMResult(req.custom_id, "error", error=type(e).__name__)
        return self.result(req.custom_id, msg, req.schema, model, batch=False)

    def submit_batch(self, reqs: Sequence[LLMRequest], *, model: str) -> tuple[str, datetime | None]:
        for r in reqs:
            if not CUSTOM_ID.match(r.custom_id):
                raise ValueError("bad custom_id")
        # Never retried by the SDK: creating a batch is not idempotent, and a retry after a timeout could create a
        # second paid batch. A lost reply is recovered by find_batch (classify.collect's watchdog).
        b = self.client.with_options(max_retries=0).messages.batches.create(
            requests=[{"custom_id": r.custom_id, "params": self.params(r, model)} for r in reqs]
        )
        return b.id, getattr(b, "expires_at", None)

    def find_batch(
        self, created_after: datetime, n_requests: int, exclude=frozenset()
    ) -> tuple[str, datetime | None] | None:
        """A batch we created but never recorded (the reply was lost): created after our row, same size, and not
        one we already track (`exclude`). `created_after` must be timezone-aware, like the SDK's timestamps."""
        for b in self.client.messages.batches.list(limit=20):
            if b.created_at < created_after:
                break  # newest first
            if b.id in exclude:
                continue
            c = b.request_counts
            if c.processing + c.succeeded + c.errored + c.canceled + c.expired == n_requests:
                return b.id, getattr(b, "expires_at", None)
        return None

    def poll_batch(self, external_id: str) -> BatchInfo:
        b = self.client.messages.batches.retrieve(external_id)
        st = {"ended": "ended", "canceling": "processing", "in_progress": "processing"}.get(b.processing_status)
        return BatchInfo(st or "failed", getattr(b, "expires_at", None))

    def fetch_results(self, external_id: str, schema, model: str) -> Iterator[LLMResult]:
        for item in self.client.messages.batches.results(external_id):
            cid = item.custom_id if CUSTOM_ID.match(item.custom_id or "") else None
            if cid is None:
                continue  # never trust an id we did not generate
            kind = item.result.type
            if kind == "succeeded":
                yield self.result(cid, item.result.message, schema, model, batch=True)
            elif kind in ("canceled", "expired"):
                yield LLMResult(cid, kind)
            else:
                yield LLMResult(cid, "error", error=getattr(getattr(item.result, "error", None), "type", "errored"))

    def cancel_batch(self, external_id: str) -> None:
        self.client.messages.batches.cancel(external_id)


_PROVIDERS = {
    "xai": lambda: OpenAICompatProvider("xai", "xai_api_key", config.XAI_BASE_URL, thinking_in_completion=False),
    "openai": lambda: OpenAICompatProvider("openai", "openai_api_key", None, thinking_in_completion=True),
    "anthropic": AnthropicProvider,
}


def get_provider(name: str):
    if name not in _PROVIDERS:
        raise KeyError(f"unknown LLM provider {name!r}")
    return _PROVIDERS[name]()


def describe(res: LLMResult) -> str:
    """One log-safe line: id, status, error class and usage. No text."""
    return json.dumps({"id": res.custom_id, "status": res.status, "error": res.error, "in": res.usage.input,
                       "cached": res.usage.cached_read, "out": res.usage.output})  # fmt: skip
