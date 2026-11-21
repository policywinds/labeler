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

"""Bill text as the labeling engine reads it: plain text from an official HTML, XML or PDF document.

Struck-out text (existing law a bill deletes) is kept and marked "[deleted: ...]"; the prompt and the quote check
both rely on that marker (policywinds_labeler.label). Bill text is untrusted input: nothing is executed, PDFs are
page-capped.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

INLINE = {"b", "i", "u", "em", "strong", "span", "a", "sup", "sub", "font", "small", "big", "s", "strike", "ins", "del"}


DELETED = ("s", "strike", "del")


class _Text(HTMLParser):
    """Text from HTML or XML. Every non-inline element boundary is a line break (CA bill XML has no
    whitespace between elements). Struck-out text (existing law the bill deletes) is kept, wrapped as
    "[deleted: ...]", so the classifier can see what a bill removes without reading it as new law."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self._del = 0

    def handle_starttag(self, tag, attrs):
        local = tag.split(":")[-1]
        if local in ("script", "style"):
            self._skip += 1
        elif local in DELETED:
            if not self._del:
                self.parts.append("[deleted: ")
            self._del += 1
        elif local not in INLINE:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        local = tag.split(":")[-1]
        if local in ("script", "style") and self._skip:
            self._skip -= 1
        elif local in DELETED and self._del:
            self._del -= 1
            if not self._del:
                self.parts.append("]")
        elif local not in INLINE:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data.replace("\n", " ") if self._del else data)


def html_to_text(raw: str) -> str:
    p = _Text()
    p.feed(raw)
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\[deleted:\s*\]", "", text)  # empty strike-outs
    text = re.sub(r"\]+(?:\s|\[)*\[deleted: ", " ", text)  # one block per run of consecutive strike-outs
    text = re.sub(r"\[+(\[deleted: )", r"\1", text)
    text = re.sub(r"\[\s*(\[deleted: [^\]]*\])\s*\]", r"\1", text)  # TX wraps strike-outs in [ ]
    text = re.sub(r"\[deleted: ([^\]]*?)\s+\]", r"[deleted: \1]", text)
    text = re.sub(r"(\[deleted: [^\]]*\])\]+", r"\1", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def extract(raw: bytes, ctype: str, state: str | None = None) -> str:
    """Plain text from an original document (HTML or PDF). NY Assembly pages: only the bill's <PRE> block."""
    if "pdf" in ctype or raw[:5] == b"%PDF-":
        return pdf_to_text(raw)
    enc = ctype.split("charset=")[-1].strip() if "charset=" in ctype else "utf-8"
    if state == "NY":
        page = raw.decode(enc, errors="replace")
        lo, hi = page.upper().find("<PRE"), page.upper().rfind("</PRE>")
        if lo < 0 or hi < lo:
            raise ValueError("no bill text block on the NY Assembly page")
        text = clean_print(html_to_text(page[lo : hi + len("</PRE>")]), "NY")
        if len(text) < 200:
            raise ValueError("NY bill text too short")
        return text
    return html_to_text(raw.decode(enc, errors="replace"))


# New York prints every page with an explanation footer, a drafting code and a running header ("A. 325--A 2"),
# which land mid-sentence in the text. Removing them keeps sentences (and quotes) whole across page breaks.
_NY_FOOTER = re.compile(
    r"\n\s*EXPLANATION--Matter in (?:italics|bold).*?(?:is old law to be omitted|omitted)\.?[^\n]*"
    r"(?:\n\s*[A-Z]{3}\d{5}-\d{2}-\d\s*)?(?=\n|$)",
    re.S,
)
_NY_HEADER = re.compile(r"\n\s*[AS]\.\s*\d+(?:--[A-Z])?\s+\d+\s*(?=\n)")


def clean_print(text: str, state: str | None) -> str:
    """Remove page furniture from printed bill text (currently New York)."""
    if state != "NY":
        return text
    text = _NY_FOOTER.sub("", text)
    return _NY_HEADER.sub("", text)


def pdf_to_text(raw: bytes, max_pages: int = 300) -> str:
    """Text layer of a PDF (untrusted input: parsed in memory, page-capped, nothing executed)."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [p.extract_text() or "" for p in reader.pages[:max_pages]]
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    if len(text.strip()) < 200:
        raise ValueError("PDF has no usable text layer")
    return text.strip()
