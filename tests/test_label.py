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

from policywinds_labeler import label, prefilter


def test_quote_verification_is_verbatim_modulo_whitespace():
    text = "SECTION 1. A person may not\n   transfer a firearm to a person younger than 21."
    ok, bad = {"quote": "may not transfer a firearm"}, {"quote": "may not sell a firearm"}
    label.verify_quotes([ok, bad], text)
    assert ok["quote_ok"] and not bad["quote_ok"]


def test_quote_verification_ignores_line_numbers_and_punctuation():
    text = "1 A person may not sell a\n2 firearm to a \u201cminor\u201d under this sec-\n3 tion."
    t = {"quote": 'A person may not sell a firearm to a "minor"'}
    label.verify_quotes([t], text)
    assert t["quote_ok"]


def test_dedupe_keeps_highest_confidence():
    out = label.dedupe([{"tag": "a/b", "confidence": 0.5}, {"tag": "a/b", "confidence": 0.9}])
    assert out == [{"tag": "a/b", "confidence": 0.9}]


def test_quote_check_survives_words_split_across_printed_lines():
    text = (
        "     1  gender affirming care means a range of social, psychological, behav-\n"
        "     2  ioral, and medical interventions designed to support a child"
    )
    t = label.verify_quotes([{"quote": "social, psychological, behavioral, and medical interventions designed"}], text)
    assert t[0]["quote_ok"]
    assert not label.verify_quotes([{"quote": "behavioral and surgical interventions"}], text)[0]["quote_ok"]


def test_custody_exclusions_and_keyword_guard():
    assert "custody-support" not in prefilter.score("State correctional facilities; visitation policies for inmates.")
    assert "custody-support" in prefilter.score("Grandparent; petition for visitation rights with a minor child.")
    tags = [{"tag": "custody-support/other"}, {"tag": "firearms/carry"}]
    assert [t["tag"] for t in label.keyword_guard(tags, {"firearms": 1})] == ["firearms/carry"]
    assert len(label.keyword_guard(tags, {"firearms": 1, "custody-support": 2})) == 2


def test_stray_foreign_script_detection():
    from types import SimpleNamespace as NS

    def res(rationale, effect):
        return NS(rationale=rationale, topics=[NS(axis_effect=effect)])

    assert label.has_stray_text(res("ok", "protecting parents from不利 rulings"))
    assert label.has_stray_text(res("Расширяет", "ok"))
    assert not label.has_stray_text(res("Señor José's bill — “quotes” and café", "§ 46.05 – fine"))


TX_REPEAL = (
    "SECTION 2. (a) A physician commits a prohibited practice if that person: (15) [deleted: performs an abortion "
    "on an unemancipated minor without the written consent of the child's parent;] [deleted: (20) otherwise "
    "performs an abortion on an unemancipated minor in violation of Chapter 33, Family Code;] SECTION 9. The "
    "following provisions are repealed: (1) Section 30.022, Civil Practice and Remedies Code; (2) Subchapter L, "
    "Chapter 74, Civil Practice and Remedies Code; (11) Chapter 170A, Health and Safety Code; SECTION 10. This Act "
    "takes effect September 1, 2027. BE IT ENACTED BY THE LEGISLATURE OF THE STATE OF TEXAS: SECTION 1."
)


def test_quote_check_accepts_list_heading_with_one_item_and_merged_strikeouts():
    quotes = {
        "list item": "The following provisions are repealed: (11) Chapter 170A, Health and Safety Code;",
        "merged strike-outs": "[deleted: (19) performs an abortion on an unemancipated minor without the written "
        "consent of the child's parent; (20) otherwise performs an abortion on an unemancipated minor]",
        "item not in list": "The following provisions are repealed: (3) Chapter 999, Penal Code of Mars;",
        "heading without a list": "BE IT ENACTED BY THE LEGISLATURE OF THE STATE OF TEXAS: (11) Chapter 170A, "
        "Health and Safety Code;",
        "invented deletion": "[deleted: (19) performs an abortion on any person at all whenever they ask]",
    }
    got = {k: label.verify_quotes([{"quote": q}], TX_REPEAL)[0]["quote_ok"] for k, q in quotes.items()}
    assert got == {"list item": True, "merged strike-outs": True, "item not in list": False,
                   "heading without a list": False, "invented deletion": False}  # fmt: skip


def test_system_prompt_is_byte_stable():
    assert label.system_prompt() == label.system_prompt()
