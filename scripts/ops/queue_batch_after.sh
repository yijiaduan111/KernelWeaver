#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../common.sh
source "$REPO_ROOT/scripts/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/ops/queue_batch_after.sh \
    --wait-output-dir PATH \
    --next-output-dir PATH \
    --next-session NAME \
    [--poll-seconds N] \
    [--reuse-output-dir] \
    -- [args passed to scripts/run_batch.sh]

Example:
  bash scripts/ops/queue_batch_after.sh \
    --wait-output-dir runs/current_run \
    --next-output-dir runs/next_run \
    --next-session kw-next \
    -- --experiment main --task-config main_l1_15
USAGE
}

WAIT_OUTPUT_DIR=""
NEXT_OUTPUT_DIR=""
NEXT_SESSION=""
POLL_SECONDS=120
ALLOW_REUSE=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-output-dir)
      WAIT_OUTPUT_DIR="${2:?missing value for --wait-output-dir}"
      shift 2
      ;;
    --next-output-dir)
      NEXT_OUTPUT_DIR="${2:?missing value for --next-output-dir}"
      shift 2
      ;;
    --next-session)
      NEXT_SESSION="${2:?missing value for --next-session}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:?missing value for --poll-seconds}"
      shift 2
      ;;
    --reuse-output-dir)
      ALLOW_REUSE=1
      shift
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[KernelWeaver] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$WAIT_OUTPUT_DIR" || -z "$NEXT_OUTPUT_DIR" || -z "$NEXT_SESSION" ]]; then
  echo "[KernelWeaver] Missing required arguments." >&2
  usage >&2
  exit 1
fi

WAIT_OUTPUT_DIR="$(kw_abs_path "$REPO_ROOT" "$WAIT_OUTPUT_DIR")"
NEXT_OUTPUT_DIR="$(kw_abs_path "$REPO_ROOT" "$NEXT_OUTPUT_DIR")"

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -lt 5 ]]; then
  echo "[KernelWeaver] --poll-seconds must be an integer >= 5." >&2
  exit 1
fi

next_output_parent="$(dirname "$NEXT_OUTPUT_DIR")"
mkdir -p "$next_output_parent"

if [[ -d "$NEXT_OUTPUT_DIR" ]]; then
  if [[ "$ALLOW_REUSE" != "1" ]] && find "$NEXT_OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "[KernelWeaver] Next output directory is not empty: $NEXT_OUTPUT_DIR" >&2
    exit 1
  fi
else
  mkdir -p "$NEXT_OUTPUT_DIR"
fi

is_waiting_batch_alive() {
  ps -eo args= | grep -F 'stark_cli.py run-kernelbench-batch' | grep -F -- "--output-dir $WAIT_OUTPUT_DIR" >/dev/null 2>&1
}

echo "[KernelWeaver] queue watcher started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[KernelWeaver] wait_output_dir=$WAIT_OUTPUT_DIR"
echo "[KernelWeaver] next_output_dir=$NEXT_OUTPUT_DIR"
echo "[KernelWeaver] next_session=$NEXT_SESSION"
echo "[KernelWeaver] poll_seconds=$POLL_SECONDS"
printf '[KernelWeaver] queued run args:'
printf ' %q' "${FORWARD_ARGS[@]}"
printf '\n'

while is_waiting_batch_alive; do
  echo "[KernelWeaver] waiting... $(date '+%Y-%m-%d %H:%M:%S')"
  sleep "$POLL_SECONDS"
done

echo "[KernelWeaver] detected current batch finished at $(date '+%Y-%m-%d %H:%M:%S')"

LAUNCH_CMD=(
  bash "$REPO_ROOT/scripts/run_batch.sh"
  --detach
  --session-name "$NEXT_SESSION"
  --output-dir "$NEXT_OUTPUT_DIR"
)

if [[ "$ALLOW_REUSE" == "1" ]]; then
  LAUNCH_CMD+=(--reuse-output-dir)
fi

LAUNCH_CMD+=("${FORWARD_ARGS[@]}")

printf '[KernelWeaver] launching queued batch:'
printf ' %q' "${LAUNCH_CMD[@]}"
printf '\n'
"${LAUNCH_CMD[@]}"
