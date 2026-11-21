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

from openai.lib._pydantic import to_strict_json_schema

from policywinds_labeler.llm import prices
from policywinds_labeler.llm.schema import Classification
from policywinds_labeler.llm.types import Usage
from policywinds_labeler.topics import DIRECTION_AXES, TOPICS, all_tags


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_schema_is_strict_and_uses_supported_keywords():
    schema = to_strict_json_schema(Classification)
    # Keywords xAI documents as best-effort or rejected (docs.x.ai structured outputs, 2026-11-11).
    banned = {"not", "if", "then", "else", "minContains", "maxContains", "pattern"}
    for node in _walk(schema):
        assert not banned & node.keys()
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
        if "maxLength" in node:
            assert node["maxLength"] <= 2048
        if "maxItems" in node:
            assert node["maxItems"] <= 256
        if "enum" in node:
            assert node["enum"]


def test_tag_enum_covers_all_topics():
    tags = all_tags()
    assert len(tags) == sum(len(v) for v in TOPICS.values())
    assert {t.split("/")[0] for t in tags} == {
        "firearms",
        "transgender",
        "abortion",
        "immigration-enforcement",
        "custody-support",
    }


def test_direction_is_a_required_enum_on_every_topic_hit():
    schema = to_strict_json_schema(Classification)
    hit = next(n for n in _walk(schema) if n.get("properties", {}).get("tag"))
    assert "direction" in hit["required"]
    values = next(n["enum"] for n in _walk(schema) if n.get("enum") and "expands" in n["enum"])
    assert set(values) == {"expands", "restricts", "mixed", "neutral"}
    assert DIRECTION_AXES.keys() == TOPICS.keys()


def test_cost_counts_cached_writes_output_and_batch():
    m = ("xai", "grok-4.20-0309-non-reasoning")
    assert round(prices.cost(*m, Usage(input=1_000_000)), 4) == 1.25
    assert round(prices.cost(*m, Usage(input=1_000_000, cached_read=1_000_000)), 4) == 0.20
    s = ("anthropic", "claude-sonnet-5-5")
    full = prices.cost(*s, Usage(input=1_000_000, cache_write=1_000_000, output=1_000_000))
    assert round(full, 4) == 2.5 + 10.0
    assert round(prices.cost(*s, Usage(input=1_000_000, output=1_000_000), batch=True), 4) == 6.0
    # the estimate covers a typical answer (Sonnet used at most 915 output tokens in the 2026-11-21 test)
    assert prices.estimate(*s, 30_000, batch=True) > prices.cost(*s, Usage(input=7_500, output=915), batch=True)


def test_unknown_model_has_no_price():
    import pytest

    with pytest.raises(KeyError):
        prices.cost("anthropic", "claude-unknown", Usage(input=1))
