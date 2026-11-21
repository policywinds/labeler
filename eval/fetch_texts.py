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

"""Download the eval texts that are not stored in this repository, from the legislature's own site.

Run from anywhere: python fetch_texts.py   (needs only the policywinds_labeler package)

bills.jsonl says how each bill's text is provided:
- "committed": the text is in texts/ already (official state, California and GovInfo documents).
- "fetch": downloaded here from the official URL and extracted the same way the labeler extracts it.
- "manual": the state publishes no document that can be fetched directly. Open the URL in a browser, copy the
  bill text and save it as the file named in the record. run_eval.py skips the bill until the file exists.

Downloads are spaced 3 seconds apart, never retried in a loop, and only made when the file is missing.
"""

import json
import sys
import time
from pathlib import Path

import httpx

from policywinds_labeler.text import extract

HERE = Path(__file__).resolve().parent
USER_AGENT = "policywinds-labeler-eval (+https://github.com/policywinds/labeler)"
SPACING_S = 3.0
MAX_BYTES = 5_000_000


def main() -> int:
    bills = [json.loads(line) for line in (HERE / "bills.jsonl").open()]
    missing = 0
    for b in bills:
        t = b["text"]
        path = HERE / t["file"]
        if t["how"] == "committed" or path.exists():
            continue
        name = f"{b['state']} {b['number']} ({b['year']})"
        if t["how"] == "manual":
            print(f"manual: {name}: save the text from {t['url']} as {t['file']}")
            missing += 1
            continue
        r = httpx.get(t["url"], timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=False)
        if r.status_code != 200 or len(r.content) > MAX_BYTES:
            print(f"failed: {name}: HTTP {r.status_code} from {t['url']}")
            missing += 1
        else:
            ctype = (r.headers.get("content-type") or "text/html").split(";")[0].strip().lower()
            if ctype == "text/html" and r.encoding:
                ctype = f"text/html; charset={r.encoding}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(extract(r.content, ctype, b["state"]))
            print(f"fetched: {name}")
        time.sleep(SPACING_S)
    print(f"{missing} text(s) still missing" if missing else "all texts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
