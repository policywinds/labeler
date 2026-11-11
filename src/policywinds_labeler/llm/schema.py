"""Structured-output schema for the multi-topic classifier (one call, all topics)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from lawtracker.topics import all_tags

Tag = Enum("Tag", {t.replace("/", "__").replace("-", "_").upper(): t for t in all_tags()}, type=str)


class TopicHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: Tag = Field(description="topic/subtag")
    confidence: float = Field(ge=0, le=1)
    quote: str = Field(max_length=600, description="Verbatim excerpt from the provided bill text supporting this tag.")
    section_ref: str | None = Field(description="Section number/heading of the quote, or null if unknown.")


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevant: bool
    topics: list[TopicHit] = Field(max_length=8)
    rationale: str = Field(max_length=600)
