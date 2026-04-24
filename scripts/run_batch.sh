#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck source=common.sh
source "$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_batch.sh [options] [extra CLI args]

Options:
  --experiment NAME         Experiment name, default: main
  --task-config NAME        Override the task manifest profile
  --output-dir PATH         Output directory, default: runs/<experiment>_<timestamp>
  --python PATH             Python executable, default: python
  --detach                  Start in a tmux session and write logs to launcher.log
  --session-name NAME       tmux session name for --detach mode
  --reuse-output-dir        Reuse an existing non-empty output directory
  -h, --help                Show this help

Environment helpers:
  KERNELWEAVER_CONDA_ENV    Conda env name to activate before launch
  KERNELWEAVER_CONDA_PREFIX Conda env prefix to activate before launch
  KERNELWEAVER_CONDA_SH     Path to conda.sh if it is not under \$HOME/miniconda3

Notes:
  - This wrapper avoids 'conda run ... | tee ...' because that pattern is brittle for long CUDA jobs.
  - Use a fresh output directory for each run to avoid mixing logs and results.
EOF
}

REPO_ROOT="$(kw_repo_root "$SCRIPT_PATH")"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-main}"
TASK_CONFIG="${TASK_CONFIG:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${EXPERIMENT}_${TIMESTAMP}}"
DETACH=0
SESSION_NAME=""
ALLOW_REUSE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT="${2:?missing value for --experiment}"
      shift 2
      ;;
    --task-config)
      TASK_CONFIG="${2:?missing value for --task-config}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --session-name)
      SESSION_NAME="${2:?missing value for --session-name}"
      shift 2
      ;;
    --reuse-output-dir)
      ALLOW_REUSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OUTPUT_DIR="$(kw_abs_path "$REPO_ROOT" "$OUTPUT_DIR")"
kw_prepare_output_dir "$OUTPUT_DIR" "$ALLOW_REUSE"

COMMAND=("$PYTHON_BIN" -u stark_cli.py run-kernelbench-batch --experiment "$EXPERIMENT" --output-dir "$OUTPUT_DIR")
if [[ -n "$TASK_CONFIG" ]]; then
  COMMAND+=(--task-config "$TASK_CONFIG")
fi
COMMAND+=("${EXTRA_ARGS[@]}")

LAUNCH_SCRIPT="$OUTPUT_DIR/start_batch.sh"
LOG_PATH="$OUTPUT_DIR/launcher.log"
kw_write_launch_script "$LAUNCH_SCRIPT" "$REPO_ROOT" COMMAND

if [[ "$DETACH" == "1" ]]; then
  SESSION_NAME="${SESSION_NAME:-kw-$(basename "$OUTPUT_DIR")}"
  SESSION_NAME="$(kw_unique_tmux_session "$SESSION_NAME")"
  tmux new-session -d -s "$SESSION_NAME" "bash $(printf '%q' "$LAUNCH_SCRIPT") >> $(printf '%q' "$LOG_PATH") 2>&1"
  echo "[KernelWeaver] Started detached batch run."
  echo "[KernelWeaver] session: $SESSION_NAME"
  echo "[KernelWeaver] log: $LOG_PATH"
  echo "[KernelWeaver] attach: tmux attach -t $SESSION_NAME"
  echo "[KernelWeaver] tail: tail -f $LOG_PATH"
else
  exec bash "$LAUNCH_SCRIPT"
fi
