from openai.lib._pydantic import to_strict_json_schema

from lawtracker.llm.client import cost_usd
from lawtracker.llm.schema import Classification
from lawtracker.topics import DIRECTION_AXES, TOPICS, all_tags


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
    assert {t.split("/")[0] for t in tags} == {"firearms", "transgender", "abortion", "immigration-enforcement"}


def test_direction_is_a_required_enum_on_every_topic_hit():
    schema = to_strict_json_schema(Classification)
    hit = next(n for n in _walk(schema) if n.get("properties", {}).get("tag"))
    assert "direction" in hit["required"]
    values = next(n["enum"] for n in _walk(schema) if n.get("enum") and "expands" in n["enum"])
    assert set(values) == {"expands", "restricts", "mixed", "neutral"}
    assert DIRECTION_AXES.keys() == TOPICS.keys()


def test_cost_counts_cached_and_reasoning_conservatively():
    m = "grok-4.20-0309-non-reasoning"
    base = cost_usd(m, {"prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert round(base, 4) == 1.25
    cached = cost_usd(
        m, {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 1_000_000}}
    )
    assert round(cached, 4) == 0.20
    with_reasoning = cost_usd(
        m, {"prompt_tokens": 0, "completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 100}}
    )
    assert with_reasoning == 200 * 2.50 / 1_000_000
