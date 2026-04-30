#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kokoro presubmit script for antigravity-sdk-py.
# Runs unit tests and linting on every GoB change.

set -eo pipefail

cd "${KOKORO_ARTIFACTS_DIR}/git/antigravity-sdk-py"

echo "--- Setting up Python environment ---"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "--- Installing package and test dependencies ---"
pip install -e ".[dev]"

echo "--- Running tests ---"
python -m pytest tests/ -v --tb=short

echo "--- Running lint ---"
python -m ruff check .

echo "--- Presubmit passed ---"
