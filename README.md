# KernelWeaver

KernelWeaver is a clean STARK-style baseline for kernel optimization experiments.
It is organized for group collaboration: easy to configure, easy to run, and easy to extend.

## What To Prepare First

### 1. Prepare your environment

Use your own Python or Conda environment.
The project needs at least:
- Python 3.10+
- `PyYAML`
- `torch`
- `triton` if you run Triton tasks
- the official `KernelBench` package path available on disk

### 2. Prepare KernelBench

Clone KernelBench to your own machine or server, then edit:
- `configs/runtime/kernelbench_paths.yaml`

Set:
- `kernelbench_root: /path/to/KernelBench`

### 3. Prepare cudaLLM weights if you use local code generation

Edit:
- `configs/models/providers/local-cudallm.yaml`

Set:
- `model_path: /path/to/cudaLLM-8B`

### 4. Prepare your own API

Do not use someone else's API.

Copy:
- `.env.example` -> `.env`

Then fill in your own values:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` if needed
- `OPENAI_MODEL` if needed

## The Four Things You Usually Choose

For one experiment, we mainly decide:
- `tasks`
- `backend`
- `route`
- `profile`

The `profile` is always one of:
- `quick`: smoke check
- `paper`: paper-style setting
- `main`: our regular setting

And each `profile` only expands into:
- `search`
- `evaluator`
- `measurement`

## Recommended Defaults

The regular group baseline is:
- experiment: `main`
- tasks: `main_l1_15`
- backend: `cuda`
- route: `codeagent_cudallm`

That means:
- `PlanAgent` -> GPT API
- `CodeAgent` -> local `cudaLLM`
- `DebugAgent` -> GPT API
- `search-agent` path -> local `cudaLLM`

## Repository Layout

- `src/`
  - `core/`: bridge, workflow, tree, context
  - `agents/`: plan, code, and debug agents
  - `providers/`: provider implementations and role routing
  - `evaluation/`: demo evaluator, paper evaluator, validation
  - `experiment/`: manifest loading and report helpers
  - `cli.py`: main CLI
  - `config.py`: layered YAML loader
  - `models.py`: shared dataclasses
  - `io.py`: run save/load helpers
- `configs/`
  - `experiments/`: top-level experiment presets
  - `tasks/`: task manifests
  - `models/providers/`: provider connection defaults
  - `models/routes/`: plan/code/debug/search routing
  - `search/`: search-tree settings
  - `evaluation/evaluators/`: evaluator path and reference modes
  - `evaluation/measurement/`: correctness and timing budgets
  - `runtime/`: paths, GPU visibility, env file
- `scripts/`: thin helper scripts
- `docs/`: onboarding and architecture notes
- `tests/`: small regression tests
- `runs/`: outputs, not for Git
- `stark/`: compatibility package so `import stark...` still works

## Stable Launch Rule

For long CUDA experiments, do not launch with:
- `conda run ...`
- `... | tee ...`
- remote wrappers that keep a fragile pipe open

Use the helper scripts in `scripts/` instead. They:
- keep one fresh output directory per run
- use `python -u`
- write a reproducible start script into the run directory
- support `tmux` detached launch for long jobs

If you want the script to activate a Conda env for you, set one of:
- `KERNELWEAVER_CONDA_ENV=stark`
- `KERNELWEAVER_CONDA_PREFIX=/path/to/env`

## Most Common Commands

### Run the regular baseline batch in tmux

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_batch.sh \
  --detach \
  --experiment main \
  --output-dir runs/main_cuda_cudallm
```

Then watch the log:

```bash
tail -f runs/main_cuda_cudallm/launcher.log
```

### Run one KernelBench task

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_single.sh \
  --experiment main \
  --level 1 \
  --problem-id 25 \
  --output-dir runs/l1_p25
```

### Run the smoke setting

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_batch.sh \
  --experiment quick \
  --output-dir runs/quick
```

### Run the paper-style setting

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_batch.sh \
  --experiment paper \
  --output-dir runs/paper
```

### Pass extra CLI overrides

```bash
KERNELWEAVER_CONDA_ENV=stark \
bash scripts/run_batch.sh \
  --detach \
  --experiment main \
  --output-dir runs/main_triton \
  --backend triton \
  --route-config codeagent_cudallm
```

### Inspect one run

```bash
python stark_cli.py show-run runs/l1_p25/run.json
```

### Verify one run again

```bash
python stark_cli.py verify-kernelbench runs/l1_p25/run.json
```

### Build one summary report

```bash
python stark_cli.py report-paper runs/main --output-dir runs/paper_report
```

## Notes

- `quick` is only for checking that the chain still works.
- `paper` is for a more paper-aligned setting.
- `main` is the default setting for our own experiments.
- New users should start from `configs/experiments/` and only then drill into lower layers.
- Use a new `runs/...` directory for each launch unless you really mean to reuse an old directory.
