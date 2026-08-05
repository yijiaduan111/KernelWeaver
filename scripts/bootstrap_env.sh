#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${KW_ENV_NAME:-kernelweaver}"
CONDA_BIN="${CONDA_BIN:-conda}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${KW_ENV_LOCK:-$REPO_ROOT/environment.lock.yml}"
BASE_FILE="$REPO_ROOT/environment.yml"

log() {
  printf '[bootstrap_env] %s\n' "$*"
}

fail() {
  printf '[bootstrap_env] ERROR: %s\n' "$*" >&2
  exit 1
}

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  fail "conda not found. Install Miniconda/Mambaforge first, or set CONDA_BIN=/path/to/conda."
fi

if [ -f "$LOCK_FILE" ]; then
  ENV_FILE="$LOCK_FILE"
else
  ENV_FILE="$BASE_FILE"
fi
[ -f "$ENV_FILE" ] || fail "missing environment file: $ENV_FILE"

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists; skipping creation"
else
  log "creating conda env '$ENV_NAME' from $(basename "$ENV_FILE")"
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

log "installing repository in editable mode"
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

log "checking core Python/CUDA packages"
"$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY'
import sys
print('python:', sys.executable)
try:
    import torch
    print('torch:', torch.__version__)
    print('torch cuda runtime:', torch.version.cuda)
    print('torch cuda available:', torch.cuda.is_available())
    print('torch cuda device count:', torch.cuda.device_count())
except Exception as exc:
    print('torch import/check failed:', type(exc).__name__, exc)
    raise SystemExit(1)
try:
    import yaml
    print('PyYAML:', yaml.__version__)
except Exception as exc:
    print('PyYAML import failed:', type(exc).__name__, exc)
    raise SystemExit(1)
PY

log "done. Activate with: conda activate $ENV_NAME"
