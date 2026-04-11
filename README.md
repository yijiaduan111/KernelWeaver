# KernelWeaver


## Default Baseline

The recommended default experiment is `paper_mini`.

It means:
- backend: `cuda`
- workflow: `stark`
- task set: `kb9_cuda`
- search budget: `10` attempts per task
- provider routing:
  - `PlanAgent` -> GPT API
  - `CodeAgent` -> local `cudaLLM`
  - `DebugAgent` -> GPT API
  - `search-agent` path -> local `cudaLLM`

The config file is:
- `configs/experiments/paper_mini.yaml`

## Environment Setup

### create a clean `kernelweaver` environment

```bash
conda env create -f environment.yml
conda activate kernelweaver
```

## Before You Run Anything

### 1. Prepare KernelBench

Put your KernelBench repo somewhere on your own machine or server.
Then edit:
- `configs/runtime/kernelbench_paths.yaml`

Set:
- `kernelbench_root: /path/to/KernelBench`

### 2. Prepare cudaLLM weights

Put your full-weight `cudaLLM-8B` directory somewhere on your own machine or server.
Then edit:
- `configs/models/providers/local-cudallm.yaml`

Set:
- `model_path: /path/to/cudaLLM-8B`

### 3. Configure your own API

Copy:
- `.env.example` -> `.env`

Then fill in your own values:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` if needed
- `OPENAI_MODEL` if needed

Do not use someone else's API key.
Each user should configure their own `.env` file.

## Repo Layout

- `src/`
  - `core/`: bridge, workflow, tree, context
  - `agents/`: plan/code/debug wrappers
  - `providers/`: model backends and role routing
  - `evaluation/`: local evaluator, paper evaluator, validation
  - `experiment/`: batch summary and report helpers
  - `cli.py`: main CLI
  - `config.py`: layered YAML config loader
  - `models.py`: core dataclasses
  - `io.py`: run artifact save and load helpers
- `configs/`: all formal configs
- `scripts/`: thin helper scripts
- `docs/`: supporting docs
- `tests/`: minimal regression tests
- `runs/`: outputs, not for Git
- `stark/`: compatibility package so `import stark...` still works

## Main Config Layers

- `configs/experiments/`: top-level experiment presets
- `configs/tasks/`: task manifests
- `configs/models/providers/`: provider connection defaults
- `configs/models/routes/`: per-role routing
- `configs/search/`: search budget and temperatures
- `configs/evaluation/evaluators/`: evaluator kind
- `configs/evaluation/measurement/`: warmup and trial counts
- `configs/runtime/`: paths, GPU visibility, env file

## Most Common Commands

### Run the default 9-task CUDA baseline

```bash
python stark_cli.py run-kernelbench-batch \
  --experiment paper_mini \
  --output-dir runs/paper_mini
```

### Run one KernelBench task with the default baseline

```bash
python stark_cli.py run-kernelbench \
  --experiment paper_mini \
  --level 1 \
  --problem-id 25 \
  --output-dir runs/l1_p25
```

### Run the smoke preset

```bash
python stark_cli.py run-kernelbench-batch \
  --experiment quick_local \
  --output-dir runs/quick_local
```

### Run the full paper-style preset

```bash
python stark_cli.py run-kernelbench-batch \
  --experiment paper_full \
  --output-dir runs/paper_full
```

### Inspect one run

```bash
python stark_cli.py show-run runs/l1_p25/run.json
```

### Re-validate one run

```bash
python stark_cli.py verify-kernelbench runs/l1_p25/run.json
```

### Build a paper-style summary report

```bash
python stark_cli.py report-paper runs/paper_mini --output-dir runs/paper_report
```

## Notes

- `quick_local` is for fast smoke testing.
- `paper_mini` is the default recommended baseline.
- `paper_full` is the heavier paper-style preset.
- New users should start from `configs/experiments/` instead of editing low-level configs first.
- On our own server, we usually reuse the `stark` environment to save space.
- On a new machine, we recommend creating a fresh `kernelweaver` environment from `environment.yml`.
