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

from policywinds_labeler import label, text


def test_struck_text_is_kept_and_marked_as_deleted():
    raw = (
        "<p>Sec. 1. [<s>RESTRICTIONS ON</s>] AWARDS must be made[<s>:</s>]</p>"
        "<p>[<s>(A) public entities;</s>]</p><p>[<s>(B) others</s>]<u>to eligible entities</u></p>"
        "<html:strike>old CA text</html:strike> new CA text<script>x()</script>"
    )
    t = text.html_to_text(raw)
    assert "[deleted: RESTRICTIONS ON] AWARDS" in t
    assert "made[deleted: : (A) public entities; (B) others]to eligible entities" in t
    assert "[deleted: old CA text] new CA text" in t and "x()" not in t and "[[" not in t


def test_ny_page_furniture_removed_and_amended_quotes_accepted():
    raw = (
        "19 twenty-one years of age, who entered the United States within the previ-\n"
        "EXPLANATION--Matter in italics (underscored) is new; matter in brackets\n"
        " is old law to be omitted.\nLBD00994-03-6\nA. 325--A 2\n"
        "1 ous five years with respect to benefits"
    )
    t = text.clean_print(raw, "NY")
    assert "EXPLANATION" not in t and "LBD00994" not in t and "A. 325" not in t
    q = [{"quote": "who entered the United States within the previous five years with respect to benefits"}]
    assert label.verify_quotes(q, t)[0]["quote_ok"]
    assert text.clean_print(raw, "TX") == raw
    amended = "knowing that [deleted: he or she] such person is prohibited by law from possessing a firearm"
    q2 = [{"quote": "knowing that such person is prohibited by law from possessing a firearm"}]
    assert label.verify_quotes(q2, amended)[0]["quote_ok"]


def test_html_to_text_drops_scripts_and_entities():
    assert text.html_to_text("<p>A&nbsp;b</p><script>x()</script><p>c</p>") == "A b\nc"


def test_ny_assembly_page_keeps_only_the_bill_block():
    page = (
        b"<html><body><div>Summary and menus</div><STYLE>U {color: Green}</STYLE><PRE WIDTH=108>"
        b"AN ACT to amend the penal law, in relation to firearms\n" + b"filler words " * 20 + b"\n"
        b"1 Section 1. A person shall [<B><S>not</S></B>] <B><U>never</U></B> carry.</PRE>"
        b"<div>Footer</div></body></html>"
    )
    t = text.extract(page, "text/html; charset=utf-8", "NY")
    assert t.startswith("AN ACT") and "[deleted: not]" in t and "Summary" not in t and "Footer" not in t
    assert "color" not in t
