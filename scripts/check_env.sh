#!/usr/bin/env bash
set -uo pipefail

ENV_NAME="${KW_ENV_NAME:-kernelweaver}"
CONDA_BIN="${CONDA_BIN:-conda}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0
TMP_FILES=()

cleanup() {
  rm -f "${TMP_FILES[@]:-}" /tmp/kw_check_out.$$ /tmp/kw_check_err.$$
}
trap cleanup EXIT

section() {
  printf '\n=== %s ===\n' "$*"
}

check() {
  local name="$1"
  shift
  printf '[check] %s ... ' "$name"
  if "$@" >/tmp/kw_check_out.$$ 2>/tmp/kw_check_err.$$; then
    printf 'ok\n'
    sed -n '1,40p' /tmp/kw_check_out.$$
  else
    printf 'FAILED\n'
    sed -n '1,80p' /tmp/kw_check_out.$$
    sed -n '1,80p' /tmp/kw_check_err.$$
    FAILURES=$((FAILURES + 1))
  fi
  rm -f /tmp/kw_check_out.$$ /tmp/kw_check_err.$$
}

write_python_checks() {
  PY_CORE_CHECK="$(mktemp /tmp/kw_core_check.XXXXXX.py)"
  PY_IMPORT_CHECK="$(mktemp /tmp/kw_import_check.XXXXXX.py)"
  TMP_FILES+=("$PY_CORE_CHECK" "$PY_IMPORT_CHECK")
  cat > "$PY_CORE_CHECK" <<'PY'
import sys
print('python:', sys.executable)
print('python version:', sys.version.replace('\n', ' '))
import yaml
print('yaml:', yaml.__version__)
import torch
print('torch:', torch.__version__)
print('torch cuda runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('torch.cuda.is_available() is false')
PY
  cat > "$PY_IMPORT_CHECK" <<'PY'
import src
import stark
print('imports ok')
PY
}

section "Repository"
printf 'repo: %s\n' "$REPO_ROOT"
(cd "$REPO_ROOT" && git branch --show-current 2>/dev/null || true)
(cd "$REPO_ROOT" && git log -1 --oneline --decorate 2>/dev/null || true)

section "Conda"
if command -v "$CONDA_BIN" >/dev/null 2>&1; then
  "$CONDA_BIN" --version
  "$CONDA_BIN" env list | sed -n '1,40p'
else
  printf 'conda not found; set CONDA_BIN=/path/to/conda\n'
  FAILURES=$((FAILURES + 1))
fi

section "Python package environment"
if command -v "$CONDA_BIN" >/dev/null 2>&1 && "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  write_python_checks
  check "python/torch/yaml" "$CONDA_BIN" run -n "$ENV_NAME" python "$PY_CORE_CHECK"
  check "kernelweaver import" env PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$CONDA_BIN" run -n "$ENV_NAME" python "$PY_IMPORT_CHECK"
else
  printf "conda env '%s' not found\n" "$ENV_NAME"
  FAILURES=$((FAILURES + 1))
fi

section "CUDA system tools"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  printf 'nvidia-smi not found\n'
  FAILURES=$((FAILURES + 1))
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 4
else
  printf 'nvcc not found in PATH; CUDA extension builds may fail unless torch extension tooling can locate CUDA_HOME\n'
fi
if command -v ncu >/dev/null 2>&1; then
  ncu --version | sed -n '1,8p'
else
  printf 'ncu not found in PATH; diagnostics NCU profiling will be unavailable\n'
fi

section "Provider environment variables"
for var in OPENAI_API_KEY OPENAI_BASE_URL CLAUDE_API_KEY CLAUDE_BASE_URL GEMINI_API_KEY GEMINI_BASE_URL; do
  if [ -n "${!var:-}" ]; then
    printf '%s=set\n' "$var"
  else
    printf '%s=missing\n' "$var"
  fi
done

section "Result"
if [ "$FAILURES" -eq 0 ]; then
  printf 'environment check passed\n'
else
  printf 'environment check finished with %s hard failure(s)\n' "$FAILURES"
fi
exit "$FAILURES"
