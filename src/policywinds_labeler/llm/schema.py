# Copyright 2026 Kernelcraft LLC (PolicyWinds)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured-output schema for the multi-topic classifier (one call, all topics)."""

from __future__ import annotations

from enum import Enum, StrEnum

from pydantic import BaseModel, ConfigDict, Field

from policywinds_labeler.topics import all_tags


class Direction(StrEnum):
    """Effect of the tagged provisions on the topic's axis (policywinds_labeler.topics.DIRECTION_AXES)."""

    EXPANDS = "expands"
    RESTRICTS = "restricts"
    MIXED = "mixed"  # provisions move the axis both ways
    NEUTRAL = "neutral"  # procedural, study, reporting or funding-only; no change on the axis


Tag = Enum("Tag", {t.replace("/", "__").replace("-", "_").upper(): t for t in all_tags()}, type=str)


class TopicHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag: Tag = Field(description="topic/subtag")
    axis_effect: str = Field(
        max_length=600,  # v9: was 300, which cut about 1 in 6 lines mid-phrase; the prompt asks for under ~200
        description="One sentence, ideally under 200 characters, written before choosing direction: does this make "
        "the topic's 'expands' outcome or its 'restricts' outcome more likely in practice, and why.",
    )
    direction: Direction = Field(description="Effect on the topic's direction axis, consistent with axis_effect.")
    confidence: float = Field(ge=0, le=1)
    quote: str = Field(max_length=600, description="Verbatim excerpt from the provided bill text supporting this tag.")
    section_ref: str | None = Field(description="Section number/heading of the quote, or null if unknown.")


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevant: bool
    topics: list[TopicHit] = Field(max_length=8)
    rationale: str = Field(max_length=600)
