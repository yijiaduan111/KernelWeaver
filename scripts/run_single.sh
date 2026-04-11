#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-quick_local}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-25}"
BACKEND="${BACKEND:-triton}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/single}"

exec "$PYTHON_BIN" stark_cli.py run-kernelbench   --experiment "$EXPERIMENT"   --level "$LEVEL"   --problem-id "$PROBLEM_ID"   --backend "$BACKEND"   --output-dir "$OUTPUT_DIR"   "$@"
