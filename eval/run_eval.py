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

"""Run the labeling prompt on the eval set (labels.jsonl) and score the answers. Costs a few cents per run.

Run from anywhere: python run_eval.py [--provider P --model M]
Keys come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY or XAI_API_KEY). Run fetch_texts.py first;
bills whose text file is missing are reported as SKIP and not scored.

Each line of labels.jsonl is one check: a bill (state, number, year), a tag and the expected direction
("expands", "restricts", "neutral", or "absent" when the bill must not get any tag of that topic).
A check passes when the exact tag has that direction or, if the model filed the bill under another subtag of
the same topic, when that topic's non-neutral directions all agree with the check.
"Real errors" count only a wrong direction, or a direction where none belongs (and the reverse): "no change"
and "not this topic" are the same answer, because both mean the bill does not move the measure.
Any model change must pass this first.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from policywinds_labeler import prefilter
from policywinds_labeler.label import PROMPT_VERSION, dedupe, keyword_guard, system_prompt, user_message, verify_quotes
from policywinds_labeler.llm.providers import get_provider
from policywinds_labeler.llm.schema import Classification
from policywinds_labeler.llm.types import LLMRequest

HERE = Path(__file__).resolve().parent
DEFAULT_PROVIDER, DEFAULT_MODEL = "anthropic", "claude-sonnet-5-5"  # what policywinds.com labels with


def load() -> tuple[list[dict], dict]:
    labels = [json.loads(line) for line in (HERE / "labels.jsonl").open()]
    bills = {(b["state"], b["number"], b["year"]): b for b in map(json.loads, (HERE / "bills.jsonl").open())}
    return labels, bills


def label_bill(provider, model: str, sysmsg: str, bill: dict) -> tuple[list[dict], float, float, str]:
    text = (HERE / bill["text"]["file"]).read_text()
    t0 = time.monotonic()
    res = provider.run_one(LLMRequest("b0d0v0", sysmsg, user_message(bill, text), Classification), model=model)
    if res.status != "ok":
        return [], res.cost_usd, time.monotonic() - t0, res.status
    tops = [h.model_dump(mode="json") for h in res.parsed.topics]
    matched = prefilter.score(f"{bill['title']} {bill.get('description') or ''}")
    return verify_quotes(keyword_guard(dedupe(tops), matched), text), res.cost_usd, time.monotonic() - t0, "ok"


def score(lab: dict, tops: list[dict]) -> tuple[bool, bool, list[dict]]:
    """(pass, real error, this topic's tags)"""
    topic = lab["tag"].split("/")[0]
    same = [t for t in tops if t["tag"].split("/")[0] == topic]
    exact = [t["direction"] for t in same if t["tag"] == lab["tag"]]
    pool = exact or [t["direction"] for t in same if t["direction"] != "neutral"] or [t["direction"] for t in same]
    want = lab["direction"]
    hit = not same if want == "absent" else bool(pool) and set(pool) == {want}
    moved = {t["direction"] for t in same if t["direction"] != "neutral"}
    if want in ("absent", "neutral"):
        real = bool(moved)
    else:
        real = not ((set(exact) == {want}) if exact and exact != ["neutral"] else moved == {want})
    return hit, real, same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    provider, sysmsg = get_provider(args.provider), system_prompt()
    labels, bills = load()
    keys = sorted({(lab["state"], lab["number"], lab["year"]) for lab in labels})
    ready = [k for k in keys if (HERE / bills[k]["text"]["file"]).exists()]
    with ThreadPoolExecutor(4) as ex:
        runs = ex.map(lambda k: label_bill(provider, args.model, sysmsg, bills[k]), ready)
        answers = dict(zip(ready, runs, strict=True))

    ok = real_errors = skipped = 0
    for lab in labels:
        key = (lab["state"], lab["number"], lab["year"])
        if key not in answers:
            skipped += 1
            print("SKIP", lab["state"], lab["number"], lab["tag"], "no text file: run fetch_texts.py")
            continue
        tops, _, _, status = answers[key]
        hit, real, same = score(lab, tops)
        ok += hit
        real_errors += real
        tags = [f"{t['tag']}={t['direction']}" for t in same]
        print("PASS" if hit else "FAIL", lab["state"], lab["number"], lab["tag"], "want", lab["direction"], tags,
              "" if status == "ok" else status)  # fmt: skip
        if not hit:
            for t in tops:
                print("    ", t["tag"], t["direction"], t["quote_ok"], "|", t["axis_effect"][:220])
    cost = sum(a[1] for a in answers.values())
    secs = [a[2] for a in answers.values()]
    print(
        f"{args.provider} {args.model}, prompt v{PROMPT_VERSION}: {ok}/{len(labels) - skipped} passed, "
        f"{real_errors} real errors, {skipped} skipped  cost ${cost:.3f}  "
        f"avg {sum(secs) / max(1, len(secs)):.1f}s per call"
    )


if __name__ == "__main__":
    main()
