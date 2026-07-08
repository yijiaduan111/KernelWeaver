#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
# shellcheck source=common.sh
source "$(cd "$(dirname "$SCRIPT_PATH")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_isolated_batch.sh [options] [extra CLI args]

Options:
  --experiment NAME         Experiment name, default: main
  --task-config NAME        Named manifest under configs/tasks (mutually exclusive with --manifest)
  --manifest PATH           Direct YAML/JSON manifest path (mutually exclusive with --task-config)
  --backend NAME            Override backend in generated per-task manifests
  --gpus IDS                Comma-separated GPU ids, default: auto-detect all GPUs
  --max-attempts N          Override max attempts per isolated task run
  --output-dir PATH         Root output directory, default: runs/<manifest>_<experiment>_isolated_<timestamp>
  --python PATH             Python executable used for manifest splitting, default: python
  --session-prefix NAME     tmux session prefix, default: kwiso-<output-dir-name>
  --sleep-seconds N         Sleep between tasks on the same GPU worker, default: 2
  --reuse-output-dir        Reuse an existing non-empty output directory
  -h, --help                Show this help

Behavior:
  - Splits a task manifest into one single-task manifest per task.
  - Launches one tmux worker per GPU.
  - Each worker runs its assigned tasks sequentially, one process at a time.
  - Each task writes into its own output directory, so failures do not poison others.

Examples:
  bash scripts/run_isolated_batch.sh \
    --experiment main \
    --task-config main_l1_15 \
    --gpus 0,1,2,3 \
    --max-attempts 30 \
    --route-config codeagent_claude \
    --env-file /data/dyj/STARK/.env.server \
    --search-config main \
    --evaluator-config main \
    --measurement-config main \
    --deliberation-config main \
    --verbose
EOF
}

REPO_ROOT="$(kw_repo_root "$SCRIPT_PATH")"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${EXPERIMENT:-main}"
TASK_CONFIG="${TASK_CONFIG:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
GPU_LIST="${GPU_LIST:-auto}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SESSION_PREFIX="${SESSION_PREFIX:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
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
    --manifest)
      MANIFEST_PATH="${2:?missing value for --manifest}"
      shift 2
      ;;
    --backend)
      BACKEND_OVERRIDE="${2:?missing value for --backend}"
      shift 2
      ;;
    --gpus)
      GPU_LIST="${2:?missing value for --gpus}"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="${2:?missing value for --max-attempts}"
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
    --session-prefix)
      SESSION_PREFIX="${2:?missing value for --session-prefix}"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="${2:?missing value for --sleep-seconds}"
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

if [[ -n "$TASK_CONFIG" && -n "$MANIFEST_PATH" ]]; then
  echo "[KernelWeaver] Use either --task-config or --manifest, not both." >&2
  exit 1
fi

if [[ -z "$TASK_CONFIG" && -z "$MANIFEST_PATH" ]]; then
  echo "[KernelWeaver] Either --task-config or --manifest is required." >&2
  exit 1
fi

if [[ -n "$TASK_CONFIG" ]]; then
  MANIFEST_SOURCE="$REPO_ROOT/configs/tasks/$TASK_CONFIG.yaml"
  MANIFEST_LABEL="$TASK_CONFIG"
else
  MANIFEST_SOURCE="$(kw_abs_path "$REPO_ROOT" "$MANIFEST_PATH")"
  MANIFEST_LABEL="$(basename "${MANIFEST_SOURCE%.*}")"
fi

if [[ ! -f "$MANIFEST_SOURCE" ]]; then
  echo "[KernelWeaver] Manifest not found: $MANIFEST_SOURCE" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="runs/${MANIFEST_LABEL}_${EXPERIMENT}_isolated_${TIMESTAMP}"
fi
OUTPUT_DIR="$(kw_abs_path "$REPO_ROOT" "$OUTPUT_DIR")"
kw_prepare_output_dir "$OUTPUT_DIR" "$ALLOW_REUSE"

MANIFEST_DIR="$OUTPUT_DIR/manifests"
WORKER_DIR="$OUTPUT_DIR/workers"
TASK_ROOT="$OUTPUT_DIR/tasks"
mkdir -p "$MANIFEST_DIR" "$WORKER_DIR" "$TASK_ROOT"

if [[ "$GPU_LIST" == "auto" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[KernelWeaver] Could not auto-detect GPUs because nvidia-smi is unavailable." >&2
    exit 1
  fi
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[KernelWeaver] No GPUs selected." >&2
  exit 1
fi

for i in "${!GPUS[@]}"; do
  GPUS[$i]="$(printf '%s' "${GPUS[$i]}" | xargs)"
  if [[ -z "${GPUS[$i]}" ]]; then
    echo "[KernelWeaver] Empty GPU id detected in --gpus." >&2
    exit 1
  fi
done

TASK_PLAN="$OUTPUT_DIR/task_plan.tsv"
"$PYTHON_BIN" - "$MANIFEST_SOURCE" "$MANIFEST_DIR" "$TASK_PLAN" "$BACKEND_OVERRIDE" <<'PY'
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to split task manifests.") from exc

manifest_source = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
task_plan = Path(sys.argv[3])
backend_override = sys.argv[4].strip()

text = manifest_source.read_text(encoding="utf-8")
if manifest_source.suffix.lower() == ".json":
    payload = json.loads(text)
else:
    payload = yaml.safe_load(text)

if not isinstance(payload, dict):
    raise SystemExit(f"Manifest must be a mapping: {manifest_source}")

tasks = payload.get("tasks")
if not isinstance(tasks, list) or not tasks:
    raise SystemExit(f"Manifest has no tasks: {manifest_source}")

base_name = str(payload.get("name") or manifest_source.stem)
manifest_dir.mkdir(parents=True, exist_ok=True)

def slugify(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return clean or "task"

with task_plan.open("w", encoding="utf-8", newline="") as handle:
    handle.write("index\talias\tlevel\tproblem_id\tmanifest_path\ttask_slug\n")
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise SystemExit(f"Task #{idx} is not a mapping in {manifest_source}")
        task_payload = dict(task)
        if backend_override:
            task_payload["backend"] = backend_override
        alias = str(task_payload.get("alias") or f"task_{idx}")
        task_slug = slugify(alias)
        single_name = f"{base_name}_{idx:02d}_{task_slug}"
        single_manifest = manifest_dir / f"{idx:02d}_{task_slug}.yaml"
        single_manifest.write_text(
            yaml.safe_dump(
                {"name": single_name, "tasks": [task_payload]},
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        level = task_payload.get("level", "")
        problem_id = task_payload.get("problem_id", "")
        handle.write(
            f"{idx}\t{alias}\t{level}\t{problem_id}\t{single_manifest}\t{task_slug}\n"
        )
PY

TASK_COUNT=$(( $(wc -l < "$TASK_PLAN") - 1 ))
if [[ "$TASK_COUNT" -le 0 ]]; then
  echo "[KernelWeaver] Failed to split tasks from manifest: $MANIFEST_SOURCE" >&2
  exit 1
fi

declare -a WORKER_TASK_FILES=()
for i in "${!GPUS[@]}"; do
  WORKER_TASK_FILES[$i]="$WORKER_DIR/gpu${GPUS[$i]}.tsv"
  : > "${WORKER_TASK_FILES[$i]}"
done

task_index=0
while IFS=$'\t' read -r idx alias level problem_id manifest_path task_slug; do
  if [[ "$idx" == "index" ]]; then
    continue
  fi
  bucket=$((task_index % ${#GPUS[@]}))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$idx" "$alias" "$level" "$problem_id" "$manifest_path" "$task_slug" >> "${WORKER_TASK_FILES[$bucket]}"
  task_index=$((task_index + 1))
done < "$TASK_PLAN"

SESSIONS_FILE="$OUTPUT_DIR/sessions.tsv"
printf 'gpu\tsession\tworker_script\tworker_log\ttask_file\n' > "$SESSIONS_FILE"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  task_file="${WORKER_TASK_FILES[$i]}"
  if [[ ! -s "$task_file" ]]; then
    continue
  fi

  worker_script="$WORKER_DIR/gpu${gpu}.sh"
  worker_log="$WORKER_DIR/gpu${gpu}.log"

  {
    cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $(printf '%q' "$REPO_ROOT")

if [[ -n "\${KERNELWEAVER_CONDA_PREFIX:-}" || -n "\${KERNELWEAVER_CONDA_ENV:-}" ]]; then
  CONDA_SH="\${KERNELWEAVER_CONDA_SH:-\$HOME/miniconda3/etc/profile.d/conda.sh}"
  if [[ ! -f "\$CONDA_SH" ]]; then
    echo "[KernelWeaver] Could not find conda.sh at: \$CONDA_SH" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "\$CONDA_SH"
  if [[ -n "\${KERNELWEAVER_CONDA_PREFIX:-}" ]]; then
    conda activate "\${KERNELWEAVER_CONDA_PREFIX}"
  else
    conda activate "\${KERNELWEAVER_CONDA_ENV}"
  fi
fi

export PYTHONUNBUFFERED="\${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES=$(printf '%q' "$gpu")

resolve_cuda_home() {
  if [[ -n "\${CUDA_HOME:-}" && -x "\$CUDA_HOME/bin/nvcc" ]]; then
    return 0
  fi
  if [[ -n "\${CUDA_HOME:-}" && ! -x "\$CUDA_HOME/bin/nvcc" ]]; then
    unset CUDA_HOME
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="\$(command -v nvcc)"
    export CUDA_HOME="\$(cd "\$(dirname "\$nvcc_path")/.." && pwd)"
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
    return 0
  fi
  local candidate
  for candidate in /usr/local/cuda-*/bin/nvcc; do
    if [[ -x "\$candidate" ]]; then
      export CUDA_HOME="\$(cd "\$(dirname "\$candidate")/.." && pwd)"
      return 0
    fi
  done
}

resolve_cuda_home
if [[ -n "\${CUDA_HOME:-}" ]]; then
  export PATH="\$CUDA_HOME/bin:\$PATH"
  if [[ -n "\${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/lib:\$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/lib"
  fi
fi

echo "[KernelWeaver] worker_start gpu=$gpu tasks_file=$(printf '%q' "$task_file") at=\$(date '+%Y-%m-%d %H:%M:%S')"
echo "[KernelWeaver] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"
echo "[KernelWeaver] CUDA_HOME=\${CUDA_HOME:-<unset>}"
echo "[KernelWeaver] NVCC_PATH=\$(command -v nvcc || printf '%s' '<missing>')"

while IFS=\$'\\t' read -r idx alias level problem_id manifest_path task_slug; do
  [[ -n "\${idx:-}" ]] || continue
  task_output=$(printf '%q' "$TASK_ROOT")/\${idx}_\${task_slug}
  mkdir -p "\$task_output"
  COMMAND=(
    $(printf '%q' "$PYTHON_BIN")
    -u
    stark_cli.py
    run-kernelbench-batch
    --experiment
    $(printf '%q' "$EXPERIMENT")
    --manifest
    "\$manifest_path"
    --output-dir
    "\$task_output"
EOF
    if [[ -n "$MAX_ATTEMPTS" ]]; then
      printf '    --max-attempts\n    %q\n' "$MAX_ATTEMPTS"
    fi
    for item in "${EXTRA_ARGS[@]}"; do
      printf '    %q\n' "$item"
    done
    cat <<EOF
  )
  echo "[KernelWeaver] task_start idx=\$idx alias=\$alias gpu=$gpu output=\$task_output at=\$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[KernelWeaver] task_command:'
  printf ' %q' "\${COMMAND[@]}"
  printf '\n'
  set +e
  "\${COMMAND[@]}" >"\$task_output/launcher.log" 2>&1
  exit_code=\$?
  set -e
  if [[ "\$exit_code" -eq 0 ]]; then
    echo "[KernelWeaver] task_done idx=\$idx alias=\$alias status=ok at=\$(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "[KernelWeaver] task_done idx=\$idx alias=\$alias status=fail exit_code=\$exit_code at=\$(date '+%Y-%m-%d %H:%M:%S')"
  fi
  sleep $(printf '%q' "$SLEEP_SECONDS")
done < $(printf '%q' "$task_file")

echo "[KernelWeaver] worker_done gpu=$gpu at=\$(date '+%Y-%m-%d %H:%M:%S')"
EOF
  } > "$worker_script"
  chmod +x "$worker_script"

  session_name="${SESSION_PREFIX:-kwiso-$(basename "$OUTPUT_DIR")}-g${gpu}"
  session_name="$(kw_unique_tmux_session "$session_name")"
  tmux new-session -d -s "$session_name" "bash $(printf '%q' "$worker_script") >> $(printf '%q' "$worker_log") 2>&1"

  printf '%s\t%s\t%s\t%s\t%s\n' "$gpu" "$session_name" "$worker_script" "$worker_log" "$task_file" >> "$SESSIONS_FILE"
done

echo "[KernelWeaver] Started isolated batch run."
echo "[KernelWeaver] output_root: $OUTPUT_DIR"
echo "[KernelWeaver] manifest_source: $MANIFEST_SOURCE"
echo "[KernelWeaver] task_count: $TASK_COUNT"
echo "[KernelWeaver] gpu_list: ${GPUS[*]}"
echo "[KernelWeaver] sessions_file: $SESSIONS_FILE"
echo "[KernelWeaver] tasks_root: $TASK_ROOT"

