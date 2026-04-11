#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-quick_local}"
TASK_CONFIG="${TASK_CONFIG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/batch}"

ARGS=(stark_cli.py run-kernelbench-batch --experiment "$EXPERIMENT" --output-dir "$OUTPUT_DIR")
if [[ -n "$TASK_CONFIG" ]]; then
  ARGS+=(--task-config "$TASK_CONFIG")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
