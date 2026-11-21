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

"""Every Python file of the labeler (package, tests, eval harness) starts with its license header."""

from pathlib import Path

import policywinds_labeler

HEADER = (Path(__file__).read_text().split("\n\n", 1)[0]) + "\n"  # this file's own header, the reference
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).parent.name == "labeler" else Path(__file__).parents[1]


def _files() -> list[Path]:
    dirs = [Path(policywinds_labeler.__file__).parent, Path(__file__).resolve().parent]
    dirs += [d for d in (ROOT / "eval", ROOT / "tests" / "eval") if d.is_dir()]
    return sorted({p for d in dirs for p in d.rglob("*.py") if "__pycache__" not in p.parts})


def test_every_file_has_the_license_header():
    assert "Licensed under the Apache License, Version 2.0" in HEADER and "Kernelcraft LLC" in HEADER
    files = _files()
    assert len(files) >= 15
    missing = [str(p) for p in files if not p.read_text().startswith(HEADER)]
    assert not missing, missing
