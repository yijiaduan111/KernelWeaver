#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="$REPO_ROOT/runs/logs/triton_claude46_waiter.log"
POLL_SECONDS=${POLL_SECONDS:-180}
FREE_MEM_THRESHOLD_MIB=${FREE_MEM_THRESHOLD_MIB:-10}
EXCLUDE_GPUS=${EXCLUDE_GPUS:-2,3}

is_excluded() {
  local gpu=$1
  IFS=',' read -r -a EXCLUDED <<< "$EXCLUDE_GPUS"
  for item in "${EXCLUDED[@]}"; do
    if [[ "$gpu" == "$item" ]]; then
      return 0
    fi
  done
  return 1
}

find_two_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | while IFS=',' read -r idx mem util; do
        idx="${idx// /}"
        mem="${mem// /}"
        util="${util// /}"
        if is_excluded "$idx"; then
          continue
        fi
        if (( mem <= FREE_MEM_THRESHOLD_MIB )) && (( util == 0 )); then
          echo "$idx"
        fi
      done
}

launch_claude_triton() {
  local visible_gpus=$1
  export KERNELWEAVER_CONDA_ENV=stark
  export KERNELWEAVER_CONDA_SH=/data/dyj/miniconda3/etc/profile.d/conda.sh
  export CUDA_VISIBLE_DEVICES="$visible_gpus"
  export CUDALLM_DEVICE=cuda:1
  export CLAUDE_MODEL=claude-sonnet-4-6
  bash "$REPO_ROOT/scripts/run_batch.sh" \
    --detach \
    --session-name kw-main-l1-15-triton-claude46-v1 \
    --experiment main \
    --output-dir runs/main_l1_15_triton_claude46_v1 \
    --task-config main_l1_15 \
    --backend triton \
    --runtime-config gpu_multi \
    --route-config codeagent_cudallm \
    --code-provider claude-compatible \
    --env-file "$REPO_ROOT/.env.server"
}

{
  echo "[KernelWeaver] Triton Claude waiter started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[KernelWeaver] poll_seconds=$POLL_SECONDS"
  echo "[KernelWeaver] free_mem_threshold_mib=$FREE_MEM_THRESHOLD_MIB"
  echo "[KernelWeaver] exclude_gpus=$EXCLUDE_GPUS"

  while true; do
    mapfile -t FREE_GPUS < <(find_two_free_gpus)
    echo "[KernelWeaver] free_gpus=${FREE_GPUS[*]:-<none>} at $(date '+%Y-%m-%d %H:%M:%S')"
    if (( ${#FREE_GPUS[@]} >= 2 )); then
      CHOSEN="${FREE_GPUS[0]},${FREE_GPUS[1]}"
      echo "[KernelWeaver] launching Claude+Triton on GPUs $CHOSEN"
      launch_claude_triton "$CHOSEN"
      echo "[KernelWeaver] launched at $(date '+%Y-%m-%d %H:%M:%S')"
      exit 0
    fi
    sleep "$POLL_SECONDS"
  done
} | tee -a "$LOG_PATH"
