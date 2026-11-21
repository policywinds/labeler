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

from policywinds_labeler import prefilter


def test_prefilter_scores_and_bill_types():
    assert prefilter.score("Relating to background checks for private firearm transfers") == {"firearms": 1}
    assert prefilter.score("Relating to identity theft") == {}
    assert prefilter.BILL_TYPES.match("HB606") and prefilter.BILL_TYPES.match("SJR3")
    assert not prefilter.BILL_TYPES.match("HR12")
    assert prefilter.STATE_TYPES["FL"].match("H1234") and prefilter.STATE_TYPES["US"].match("HR18")


def test_windows_keep_hits_and_cap_size():
    text = "x" * 50000 + " firearm " + "y" * 50000
    w = prefilter.windows(text, radius=100, max_chars=5000)
    assert "firearm" in w and len(w) < 5000


def test_windows_keep_repeal_clauses():
    filler = "x " * 5000
    text = "AN ACT relating to firearms.\n" + filler + "SECTION 9. Chapter 2273, Government Code, is repealed.\n"
    assert "Chapter 2273, Government Code, is repealed" in prefilter.windows(text, radius=200)


def test_ny_and_va_bill_numbering():
    assert (
        prefilter.STATE_TYPES["NY"].match("A01234")
        and prefilter.STATE_TYPES["NY"].match("S7263")
        and not prefilter.STATE_TYPES["NY"].match("J1")
    )
    assert (
        prefilter.STATE_TYPES["VA"].match("HJ1")
        and prefilter.STATE_TYPES["VA"].match("SB44")
        and not prefilter.STATE_TYPES["VA"].match("HR3")
    )


def test_custody_exclusions_homeschool_records_divorce_grounds():
    for title in (
        "Prohibits school districts from authorizing a student to be home schooled if a parent has a custody order",
        "Divorce; cruelty, reasonable apprehension of bodily hurt, or willful desertion; child support unaffected.",
    ):
        assert "custody-support" not in prefilter.score(title), title
    assert "custody-support" in prefilter.score("Divorce; adultery, filing, parties living separate; spousal support.")
