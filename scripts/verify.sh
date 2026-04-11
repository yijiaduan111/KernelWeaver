#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PATH="${1:?usage: scripts/verify.sh <run.json> [extra args]}"
shift || true

exec "$PYTHON_BIN" stark_cli.py verify-kernelbench "$RUN_PATH" "$@"
