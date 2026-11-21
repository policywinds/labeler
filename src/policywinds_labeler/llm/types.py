"""Provider-neutral request and result types (docs/llm-providers-design.md section 1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

# Generated from integers only, never from text. Text-version ids are negative for texts that do not come from
# LegiScan (state sites, Congress.gov): "n" stands for the minus sign (b76706dn18619772v9).
CUSTOM_ID = re.compile(r"^b([0-9]{1,12})d(n?)([0-9]{1,12})v([0-9]{1,4})$")
STATUSES = ("ok", "refusal", "truncated", "invalid", "error", "expired", "canceled")


def custom_id(bill_id: int, doc_id: int, prompt_version: int) -> str:
    doc = int(doc_id)
    cid = f"b{int(bill_id)}d{'n' if doc < 0 else ''}{abs(doc)}v{int(prompt_version)}"
    if not CUSTOM_ID.match(cid):
        raise ValueError("bad custom_id")
    return cid


def parse_custom_id(cid: str) -> tuple[int, int, int]:
    m = CUSTOM_ID.match(cid)
    if not m:
        raise ValueError("bad custom_id")
    return int(m[1]), -int(m[3]) if m[2] else int(m[3]), int(m[4])


@dataclass(frozen=True)
class LLMRequest:
    custom_id: str
    system: str  # byte-identical across requests and always first, so providers can cache it
    user: str  # the bill-specific part only
    schema: type[BaseModel]


@dataclass
class Usage:
    input: int = 0  # all input tokens, including cached ones
    cached_read: int = 0
    cache_write: int = 0
    output: int = 0  # visible output plus thinking, as billed
    thinking: int = 0  # informational; already inside `output`


@dataclass
class LLMResult:
    custom_id: str
    status: str
    parsed: BaseModel | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    returned_model: str | None = None
    error: str | None = None  # error class or short code; never bill text or a provider message body

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}")


@dataclass(frozen=True)
class BatchInfo:
    status: str  # processing | ended | canceled | expired | failed
    expires_at: datetime | None = None
