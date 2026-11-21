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

"""The package version is written in two places; they must agree (the release workflow checks the tag against
pyproject.toml, and users read policywinds_labeler.__version__)."""

import tomllib
from pathlib import Path

import policywinds_labeler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if HERE.name == "labeler" else HERE.parent


def _pyproject() -> dict:
    for p in (ROOT / "pyproject.toml", ROOT / "labeler" / "pyproject.toml"):
        if p.exists():
            data = tomllib.loads(p.read_text())
            if data["project"]["name"] == "policywinds-labeler":
                return data
    raise AssertionError("policywinds-labeler pyproject.toml not found")


def test_package_version_matches_pyproject():
    assert policywinds_labeler.__version__ == _pyproject()["project"]["version"]


def test_prompt_version_is_exposed():
    from policywinds_labeler.label import PROMPT_VERSION

    assert isinstance(PROMPT_VERSION, int) and PROMPT_VERSION >= 1
