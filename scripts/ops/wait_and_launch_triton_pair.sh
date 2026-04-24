#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$REPO_ROOT/runs/logs/triton_pair_waiter.log"
POLL_SECONDS=${POLL_SECONDS:-180}
FREE_MEM_THRESHOLD_MIB=${FREE_MEM_THRESHOLD_MIB:-10}

find_four_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v mem_threshold="$FREE_MEM_THRESHOLD_MIB" '
      {
        idx=$1; mem=$2; util=$3;
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util);
        if ((mem + 0) <= mem_threshold && (util + 0) == 0) {
          print idx;
        }
      }
    '
}

launch_run() {
  local visible_gpus=$1
  local output_dir=$2
  local session_name=$3
  shift 3
  export KERNELWEAVER_CONDA_ENV=stark
  export KERNELWEAVER_CONDA_SH=/data/dyj/miniconda3/etc/profile.d/conda.sh
  export CUDA_VISIBLE_DEVICES="$visible_gpus"
  export CUDALLM_DEVICE=cuda:1
  bash "$REPO_ROOT/scripts/run_batch.sh" \
    --detach \
    --session-name "$session_name" \
    --experiment main \
    --output-dir "$output_dir" \
    --task-config main_l1_15 \
    --backend triton \
    --runtime-config gpu_multi \
    "$@"
}

{
  echo "[KernelWeaver] Triton pair waiter started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[KernelWeaver] poll_seconds=$POLL_SECONDS"
  echo "[KernelWeaver] free_mem_threshold_mib=$FREE_MEM_THRESHOLD_MIB"

  while true; do
    mapfile -t FREE_GPUS < <(find_four_free_gpus)
    echo "[KernelWeaver] free_gpus=${FREE_GPUS[*]:-<none>} at $(date '+%Y-%m-%d %H:%M:%S')"
    if (( ${#FREE_GPUS[@]} >= 4 )); then
      RUN1_GPUS="${FREE_GPUS[0]},${FREE_GPUS[1]}"
      RUN2_GPUS="${FREE_GPUS[2]},${FREE_GPUS[3]}"
      echo "[KernelWeaver] launching cudaLLM+Triton on GPUs $RUN1_GPUS"
      launch_run "$RUN1_GPUS" "runs/main_l1_15_triton_cudallm_v1" "kw-main-l1-15-triton-cudallm-v1" --route-config codeagent_cudallm --env-file "$REPO_ROOT/.env.server"
      echo "[KernelWeaver] launching Claude+Triton on GPUs $RUN2_GPUS"
      export CLAUDE_MODEL=claude-sonnet-4-6
      launch_run "$RUN2_GPUS" "runs/main_l1_15_triton_claude46_v1" "kw-main-l1-15-triton-claude46-v1" --route-config codeagent_cudallm --code-provider claude-compatible --env-file "$REPO_ROOT/.env.server"
      echo "[KernelWeaver] both Triton runs launched at $(date '+%Y-%m-%d %H:%M:%S')"
      exit 0
    fi
    sleep "$POLL_SECONDS"
  done
} | tee -a "$LOG_PATH"
