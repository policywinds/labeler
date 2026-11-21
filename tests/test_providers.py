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

from types import SimpleNamespace as NS

import pytest

from policywinds_labeler.llm import providers
from policywinds_labeler.llm.schema import Classification
from policywinds_labeler.llm.types import LLMRequest, LLMResult, custom_id

GOOD = '{"relevant": true, "topics": [], "rationale": "No topic."}'


def test_custom_id_is_built_from_integers_only():
    from policywinds_labeler.llm.types import parse_custom_id

    assert custom_id(12, 34, 9) == "b12d34v9"
    assert custom_id(76706, -18619772, 9) == "b76706dn18619772v9"  # non-LegiScan texts have negative ids
    assert parse_custom_id("b76706dn18619772v9") == (76706, -18619772, 9)
    with pytest.raises(ValueError):
        custom_id("1;drop", 2, 3)


def _msg(stop="end_turn", text=GOOD, cr=0, cw=0):
    return NS(stop_reason=stop, model="claude-sonnet-5-5", content=[NS(type="thinking"), NS(type="text", text=text)],
              usage=NS(input_tokens=100, cache_read_input_tokens=cr, cache_creation_input_tokens=cw, output_tokens=50))  # fmt: skip


def test_anthropic_result_mapping_and_usage():
    p = providers.AnthropicProvider()
    ok = p.result("b1d2v9", _msg(cr=3000), Classification, "claude-sonnet-5-5", batch=False)
    assert ok.status == "ok" and ok.parsed.relevant and ok.usage.input == 3100 and ok.usage.cached_read == 3000
    assert p.result("x", _msg(stop="refusal"), Classification, "claude-sonnet-5-5", batch=False).status == "refusal"
    assert (
        p.result("x", _msg(stop="max_tokens"), Classification, "claude-sonnet-5-5", batch=False).status == "truncated"
    )
    bad = p.result("x", _msg(text='{"relevant": true}'), Classification, "claude-sonnet-5-5", batch=False)
    assert bad.status == "invalid" and bad.parsed is None
    long_effect = GOOD.replace('"topics": []', '"topics": [{"tag": "firearms/carry", "axis_effect": "' + "x" * 700 +
                               '", "direction": "expands", "confidence": 0.5, "quote": "q", "section_ref": null}]')  # fmt: skip
    assert p.result("x", _msg(text=long_effect), Classification, "claude-sonnet-5-5", batch=False).status == "invalid"
    batch = p.result("x", _msg(), Classification, "claude-sonnet-5-5", batch=True)
    assert batch.cost_usd == pytest.approx(ok.cost_usd * 0 + (100 * 2 + 50 * 10) / 1e6 * 0.5)


def test_anthropic_request_caches_the_system_prompt():
    params = providers.AnthropicProvider.params(LLMRequest("b1d1v9", "SYS", "USER", Classification), "m")
    assert params["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert params["messages"] == [{"role": "user", "content": "USER"}]
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_batch_results_ignore_ids_we_did_not_make():
    p = providers.AnthropicProvider()
    items = [NS(custom_id="b1d2v9", result=NS(type="succeeded", message=_msg())),
             NS(custom_id="../etc", result=NS(type="succeeded", message=_msg())),
             NS(custom_id="b3d4v9", result=NS(type="expired"))]  # fmt: skip
    p.__dict__["client"] = NS(messages=NS(batches=NS(results=lambda _id: items)))
    got = list(p.fetch_results("msgbatch_1", Classification, "claude-sonnet-5-5"))
    assert [(r.custom_id, r.status) for r in got] == [("b1d2v9", "ok"), ("b3d4v9", "expired")]


def test_openai_compat_mapping():
    p = providers.OpenAICompatProvider("openai", "openai_api_key", None, thinking_in_completion=True)

    def resp(finish="stop", content=GOOD, refusal=None):
        usage = NS(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 40,
                                       "prompt_tokens_details": {"cached_tokens": 80},
                                       "completion_tokens_details": {"reasoning_tokens": 30}})  # fmt: skip
        return NS(model="gpt-6.2-sol", usage=usage,
                  choices=[NS(finish_reason=finish, message=NS(content=content, refusal=refusal))])  # fmt: skip

    for r, want in ((resp(), "ok"), (resp("length"), "truncated"), (resp(refusal="no"), "refusal"),
                    (resp("content_filter"), "refusal"), (resp(content="{}"), "invalid")):  # fmt: skip
        p.__dict__["client"] = NS(chat=NS(completions=NS(create=lambda r=r, **kw: r)))
        res = p.run_one(LLMRequest("b1d1v9", "S", "U", Classification), model="gpt-6.2-sol")
        assert res.status == want
    assert res.usage.output == 40 and res.usage.cached_read == 80  # OpenAI counts reasoning inside completion


def test_unknown_provider_and_status():
    with pytest.raises(KeyError):
        providers.get_provider("nope")
    with pytest.raises(ValueError):
        LLMResult("b1d1v1", "weird")
