#!/usr/bin/env bash
set -uo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sequential_tasks.sh [options] [extra run_single args]

Runs a task manifest one problem at a time. Each task is launched as an isolated
single-task run, and the next task starts only after the previous process exits.
Failures are recorded and, by default, do not stop later tasks.

Options:
  --experiment NAME         Experiment profile, default: main
  --task-config NAME        Task manifest under configs/tasks, default: main_l1_15
  --backend NAME            Backend, default: cuda
  --output-root PATH        Root output directory, default: runs/sequential_<task>_<timestamp>
  --python PATH             Python executable, default: python
  --gpu ID                  Set CUDA_VISIBLE_DEVICES and CUDALLM_DEVICE, default: keep env / 0 if unset
  --route-config NAME       Forwarded to run_single, e.g. codeagent_claude
  --deliberation-config NAME Forwarded to run_single, e.g. main
  --detach                  Run the whole sequential driver in one tmux session
  --session-name NAME       tmux session name for --detach
  --stop-on-error           Stop after the first failed task
  --continue-on-error       Continue after failed tasks, default
  --reuse-output-root       Allow existing non-empty output root
  -h, --help                Show this help

Environment helpers:
  KERNELWEAVER_CONDA_ENV    Conda env name activated by run_single.sh
  KERNELWEAVER_CONDA_PREFIX Conda env prefix activated by run_single.sh
  KERNELWEAVER_CONDA_SH     Path to conda.sh
  KERNELWEAVER_ENV_FILE     Env file passed to run_single.sh

Examples:
  export KERNELWEAVER_CONDA_SH=/data/dyj/miniconda3/etc/profile.d/conda.sh
  export KERNELWEAVER_CONDA_ENV=stark
  export KERNELWEAVER_ENV_FILE=/data/dyj/STARK/.env.server
  bash scripts/run_sequential_tasks.sh \
    --task-config main_l1_15 \
    --experiment main \
    --backend cuda \
    --route-config codeagent_claude \
    --deliberation-config main \
    --python /data/dyj/miniconda3/envs/stark/bin/python \
    --gpu 0 \
    --detach
EOF
}

repo_root="$(kw_repo_root "$SCRIPT_PATH")"
python_bin="${PYTHON_BIN:-python}"
experiment="${EXPERIMENT:-main}"
task_config="${TASK_CONFIG:-main_l1_15}"
backend="${BACKEND:-cuda}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_root=""
gpu_id="${CUDA_VISIBLE_DEVICES:-}"
route_config=""
deliberation_config=""
detach=0
session_name=""
stop_on_error=0
reuse_output_root=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      experiment="${2:?missing value for --experiment}"
      shift 2
      ;;
    --task-config)
      task_config="${2:?missing value for --task-config}"
      shift 2
      ;;
    --backend)
      backend="${2:?missing value for --backend}"
      shift 2
      ;;
    --output-root)
      output_root="${2:?missing value for --output-root}"
      shift 2
      ;;
    --python)
      python_bin="${2:?missing value for --python}"
      shift 2
      ;;
    --gpu)
      gpu_id="${2:?missing value for --gpu}"
      shift 2
      ;;
    --route-config)
      route_config="${2:?missing value for --route-config}"
      shift 2
      ;;
    --deliberation-config)
      deliberation_config="${2:?missing value for --deliberation-config}"
      shift 2
      ;;
    --detach)
      detach=1
      shift
      ;;
    --session-name)
      session_name="${2:?missing value for --session-name}"
      shift 2
      ;;
    --stop-on-error)
      stop_on_error=1
      shift
      ;;
    --continue-on-error)
      stop_on_error=0
      shift
      ;;
    --reuse-output-root)
      reuse_output_root=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$output_root" ]]; then
  output_root="runs/sequential_${task_config}_${experiment}_${backend}_${timestamp}"
fi
output_root="$(kw_abs_path "$repo_root" "$output_root")"

prepare_output_root() {
  if [[ -e "$output_root" && "$reuse_output_root" != "1" ]]; then
    if [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "[KernelWeaver] Output root already exists and is non-empty: $output_root" >&2
      echo "[KernelWeaver] Use --reuse-output-root or choose a fresh --output-root." >&2
      exit 1
    fi
  fi
  mkdir -p "$output_root"
}

manifest_path() {
  local candidate="$task_config"
  if [[ "$candidate" == *.yaml || "$candidate" == *.yml ]]; then
    if [[ -f "$candidate" ]]; then
      kw_abs_path "$repo_root" "$candidate"
      return
    fi
  fi
  if [[ -f "$repo_root/configs/tasks/${candidate}.yaml" ]]; then
    printf '%s\n' "$repo_root/configs/tasks/${candidate}.yaml"
    return
  fi
  if [[ -f "$repo_root/configs/tasks/${candidate}" ]]; then
    printf '%s\n' "$repo_root/configs/tasks/${candidate}"
    return
  fi
  echo "[KernelWeaver] Could not find task config: $task_config" >&2
  exit 1
}

load_tasks() {
  local manifest="$1"
  python3 - "$manifest" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = []
current = {}
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].rstrip()
    if not line.strip():
        continue
    alias_match = re.match(r"\s*-\s*alias:\s*(.+?)\s*$", line)
    if alias_match:
        if current:
            items.append(current)
        current = {"alias": alias_match.group(1).strip().strip('"\'')}
        continue
    key_match = re.match(r"\s*(level|problem_id):\s*(\d+)\s*$", line)
    if key_match and current is not None:
        current[key_match.group(1)] = int(key_match.group(2))
if current:
    items.append(current)
for index, item in enumerate(items, 1):
    try:
        alias = item["alias"]
        level = item["level"]
        problem_id = item["problem_id"]
    except KeyError as exc:
        raise SystemExit(f"Malformed task entry #{index}: missing {exc.args[0]} in {path}")
    print(f"{index}\t{alias}\t{level}\t{problem_id}")
PY
}

run_driver() {
  prepare_output_root
  local manifest
  manifest="$(manifest_path)"
  local driver_log="$output_root/sequential.log"
  local status_file="$output_root/status.tsv"
  local tasks_file="$output_root/tasks.tsv"
  load_tasks "$manifest" > "$tasks_file"

  {
    echo "[KernelWeaver] sequential_start=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[KernelWeaver] repo=$repo_root"
    echo "[KernelWeaver] manifest=$manifest"
    echo "[KernelWeaver] output_root=$output_root"
    echo "[KernelWeaver] experiment=$experiment backend=$backend python=$python_bin"
    echo "[KernelWeaver] route_config=${route_config:-<unset>} deliberation_config=${deliberation_config:-<unset>}"
    echo "[KernelWeaver] gpu=${gpu_id:-<env>} stop_on_error=$stop_on_error"
  } | tee -a "$driver_log"

  printf 'index\talias\tlevel\tproblem_id\tstatus\texit_code\tstarted_at\tfinished_at\toutput_dir\n' > "$status_file"

  local failures=0
  local total=0
  while IFS=$'\t' read -r index alias level problem_id; do
    [[ -n "$index" ]] || continue
    total=$((total + 1))
    local task_stamp safe_alias task_dir task_log started finished rc status
    task_stamp="$(date +%Y%m%d_%H%M%S)"
    safe_alias="$(printf '%s' "$alias" | tr -cs 'A-Za-z0-9_.-' '_')"
    task_dir="$output_root/$(printf '%02d' "$index")_${safe_alias}_P${problem_id}_${task_stamp}"
    task_log="$task_dir/launcher.log"
    mkdir -p "$task_dir"
    started="$(date '+%Y-%m-%d %H:%M:%S')"

    {
      echo "[KernelWeaver] task_start index=$index alias=$alias level=$level problem_id=$problem_id started_at=$started"
      echo "[KernelWeaver] task_output=$task_dir"
    } | tee -a "$driver_log"

    local cmd=(
      bash scripts/run_single.sh
      --experiment "$experiment"
      --level "$level"
      --problem-id "$problem_id"
      --backend "$backend"
      --output-dir "$task_dir"
      --python "$python_bin"
      --reuse-output-dir
    )
    if [[ -n "$route_config" ]]; then
      cmd+=(--route-config "$route_config")
    fi
    if [[ -n "$deliberation_config" ]]; then
      cmd+=(--deliberation-config "$deliberation_config")
    fi
    cmd+=("${extra_args[@]}")

    set +e
    (
      cd "$repo_root" || exit 1
      if [[ -n "$gpu_id" ]]; then
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export CUDALLM_DEVICE="$gpu_id"
      elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES=0
        export CUDALLM_DEVICE=0
      fi
      "${cmd[@]}"
    ) >> "$task_log" 2>&1
    rc=$?
    set -e
    finished="$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "$rc" == "0" ]]; then
      status="ok"
    else
      status="failed"
      failures=$((failures + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$index" "$alias" "$level" "$problem_id" "$status" "$rc" "$started" "$finished" "$task_dir" >> "$status_file"
    echo "[KernelWeaver] task_done index=$index alias=$alias status=$status exit_code=$rc finished_at=$finished" | tee -a "$driver_log"

    if [[ "$rc" != "0" && "$stop_on_error" == "1" ]]; then
      echo "[KernelWeaver] stop_on_error triggered at task index=$index problem_id=$problem_id" | tee -a "$driver_log"
      break
    fi
  done < "$tasks_file"

  echo "[KernelWeaver] sequential_done total=$total failures=$failures finished_at=$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$driver_log"
  echo "[KernelWeaver] status=$status_file" | tee -a "$driver_log"
  if [[ "$failures" -gt 0 ]]; then
    return 1
  fi
  return 0
}

if [[ "$detach" == "1" ]]; then
  prepare_output_root
  driver_script="$output_root/start_sequential.sh"
  log_path="$output_root/sequential.log"
  quoted_args=()
  for arg in "$@"; do
    quoted_args+=("$(printf '%q' "$arg")")
  done
  cat > "$driver_script" <<EOF
#!/usr/bin/env bash
cd $(printf '%q' "$repo_root")
exec bash $(printf '%q' "$SCRIPT_PATH") \
  --experiment $(printf '%q' "$experiment") \
  --task-config $(printf '%q' "$task_config") \
  --backend $(printf '%q' "$backend") \
  --output-root $(printf '%q' "$output_root") \
  --python $(printf '%q' "$python_bin") \
  $(if [[ -n "$gpu_id" ]]; then printf -- '--gpu %q \\\n  ' "$gpu_id"; fi)$(if [[ -n "$route_config" ]]; then printf -- '--route-config %q \\\n  ' "$route_config"; fi)$(if [[ -n "$deliberation_config" ]]; then printf -- '--deliberation-config %q \\\n  ' "$deliberation_config"; fi)$(if [[ "$stop_on_error" == "1" ]]; then printf -- '--stop-on-error \\\n  '; else printf -- '--continue-on-error \\\n  '; fi)--reuse-output-root ${extra_args[*]+"${extra_args[@]}"}
EOF
  chmod +x "$driver_script"
  session_name="${session_name:-kw-seq-${task_config}-${timestamp}}"
  session_name="$(kw_unique_tmux_session "$session_name")"
  tmux new-session -d -s "$session_name" "bash $(printf '%q' "$driver_script") >> $(printf '%q' "$log_path") 2>&1"
  echo "[KernelWeaver] Started detached sequential run."
  echo "[KernelWeaver] session: $session_name"
  echo "[KernelWeaver] output_root: $output_root"
  echo "[KernelWeaver] log: $log_path"
  echo "[KernelWeaver] status: $output_root/status.tsv"
  echo "[KernelWeaver] attach: tmux attach -t $session_name"
  echo "[KernelWeaver] tail: tail -f $log_path"
else
  run_driver
fi