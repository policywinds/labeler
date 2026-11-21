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

"""Price table per (provider, model), USD per 1M tokens. A model with no entry raises: the cost guard fails closed.

batch: multiplier on every token type for batch jobs (1.0 where no batch discount is offered or verified).
cache_write: price of input tokens written to the cache (Anthropic charges 1.25x input for the 5-minute cache;
the OpenAI-compatible providers cache automatically and bill writes as plain input).
"""

from __future__ import annotations

from policywinds_labeler.llm.types import Usage

PRICES: dict[tuple[str, str], dict] = {
    # docs.x.ai model list, checked 2026-11-21. xAI batch discount not verified: 1.0.
    ("xai", "grok-4.20-0309-non-reasoning"): {"input": 1.25, "cached_input": 0.20, "cache_write": 1.25, "output": 2.50, "batch": 1.0},
    ("xai", "grok-4.20-0309-reasoning"): {"input": 1.25, "cached_input": 0.20, "cache_write": 1.25, "output": 2.50, "batch": 1.0},
    # developers.openai.com/api/docs/pricing, checked 2026-11-21 (standard tier, short context).
    ("openai", "gpt-6.2-sol"): {"input": 2.0, "cached_input": 0.10, "cache_write": 2.0, "output": 10.0, "batch": 0.5},
    ("openai", "gpt-6.1-terra"): {"input": 1.0, "cached_input": 0.05, "cache_write": 1.0, "output": 5.0, "batch": 0.5},
    # Anthropic price list, checked 2026-11-21: cache reads 0.1x input except Sonnet 5.5 ($0.10), writes 1.25x.
    ("anthropic", "claude-sonnet-5-5"): {"input": 2.0, "cached_input": 0.10, "cache_write": 2.5, "output": 10.0, "batch": 0.5},
    ("anthropic", "claude-opus-5-5"): {"input": 4.0, "cached_input": 0.20, "cache_write": 5.0, "output": 20.0, "batch": 0.5},
    ("anthropic", "claude-haiku-5-5"): {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50, "batch": 0.5},
}  # fmt: skip


def price(provider: str, model: str) -> dict:
    return PRICES[(provider, model)]  # KeyError on purpose: no price, no spending


def cost(provider: str, model: str, u: Usage, *, batch: bool = False) -> float:
    p = price(provider, model)
    plain = max(0, u.input - u.cached_read - u.cache_write)
    usd = (plain * p["input"] + u.cached_read * p["cached_input"] + u.cache_write * p["cache_write"]
           + u.output * p["output"]) / 1_000_000  # fmt: skip
    return usd * (p["batch"] if batch else 1.0)


# Estimate for a request before it runs (budget reservation): a generous average, not a per-request ceiling
# (providers.MAX_OUTPUT_TOKENS is the ceiling). Input: ~4 characters per token in English, so 3.5 over-counts a
# little; the cached system prompt is priced at the full input rate, since caching inside a batch is not
# guaranteed. Output: Sonnet 5.5 used 100-915 tokens (median 268) on the 2026-11-21 test; 1,500 is ~1.6x the max.
EST_CHARS_PER_TOKEN = 3.5
EST_OUTPUT_TOKENS = 1500


def estimate(provider: str, model: str, chars: int, *, batch: bool) -> float:
    u = Usage(input=int(chars / EST_CHARS_PER_TOKEN) + 1, output=EST_OUTPUT_TOKENS)
    return cost(provider, model, u, batch=batch)
