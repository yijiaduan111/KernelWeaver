#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/report}"

exec "$PYTHON_BIN" stark_cli.py report-paper --output-dir "$OUTPUT_DIR" "$@"
