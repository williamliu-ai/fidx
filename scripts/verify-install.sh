#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

# Local clean-machine proof: build the wheel, then in a pristine python:slim
# container install fidx (binary-only) and run `fidx doctor` + the ~1k-doc e2e.
# Usage: scripts/verify-install.sh [PYVER ...]   (default: 3.11 3.12)
set -euo pipefail
cd "$(dirname "$0")/.."

PYVERS=("$@")
[ ${#PYVERS[@]} -eq 0 ] && PYVERS=(3.11 3.12)

echo "==> building wheel"
rm -rf dist
uv build >/dev/null
wheel=$(ls dist/*.whl)
echo "    $wheel"

for pv in "${PYVERS[@]}"; do
  echo "==> docker build (python:${pv}-slim)"
  docker build -q -f docker/Dockerfile.linux --build-arg "PYVER=${pv}" \
    -t "fidx-verify:${pv}" . >/dev/null
  echo "==> docker run — doctor + e2e on python ${pv}"
  docker run --rm "fidx-verify:${pv}"
done

echo "ALL LINUX VERIFY PASSED (${PYVERS[*]})"
