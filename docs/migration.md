# KernelWeaver Migration Guide

This document records the lightweight migration path for moving the active experiment branch to a new server without copying a full conda environment archive.

For the practical server move checklist, see `docs/server_migration_checklist.md`.

## Current baseline

- Active branch: `feature/model-deliberation-v0`
- Primary conda environment on the old server: `stark`
- Recommended new environment name: `kernelweaver`
- Core runtime snapshot: see `docs/env_snapshot.md`
- Rebuild files:
  - `environment.yml`: small hand-written baseline
  - `environment.lock.yml`: exported conda environment snapshot
  - `requirements-lock.txt`: pip package freeze from the working `stark` env

## Fast migration steps

```bash
git clone <repo-url> KernelWeaver
cd KernelWeaver
git checkout feature/model-deliberation-v0

# Option A: use the locked snapshot, default path.
bash scripts/bootstrap_env.sh

# Option B: choose a different env name.
KW_ENV_NAME=kernelweaver bash scripts/bootstrap_env.sh
```

Then copy the private runtime configuration from the old server:

```bash
cp /path/from/old/server/.env .env
```

Do not commit `.env`; it contains provider credentials and endpoint settings.

## Check the new server

```bash
KW_ENV_NAME=kernelweaver bash scripts/check_env.sh
```

The check should verify:

- conda environment exists
- Python can import `torch`, `yaml`, `src`, and `stark`
- `torch.cuda.is_available()` is true
- `nvidia-smi` can see GPUs
- `nvcc` is available if CUDA extension compilation is required
- `ncu` is available if Nsight Compute diagnostics are enabled
- API-related environment variables are set when experiments need live providers

## Smoke test after migration

Run a minimal single-task experiment before launching large batches. A practical first target is one of the small historical tasks, for example P40 with a low attempt count.

Use the same experiment command pattern as the old server, but keep the first run small enough to validate:

- API routing
- CUDA compilation
- isolated evaluator
- candidate correctness check
- optional NCU diagnostics
- result writing under `runs/`

## When lock rebuild fails

The most likely failure points are GPU driver / CUDA / PyTorch compatibility.

1. Check the new server driver:

```bash
nvidia-smi
```

2. Compare with `docs/env_snapshot.md`.
3. If PyTorch CUDA runtime is incompatible, install a matching PyTorch wheel for the new driver/CUDA stack, then run:

```bash
python -m pip install -e .
bash scripts/check_env.sh
```

## Alternative: full environment archive

For the fastest byte-for-byte migration, use `conda-pack` on the old server later:

```bash
conda install -n base conda-pack
conda pack -n stark -o kernelweaver-stark-env.tar.gz
```

This creates a large archive and is not committed to Git. It is useful when dependency rebuild is slow or package indexes drift.
