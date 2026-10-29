# KernelWeaver

PyramidKernel GPU kernel optimization with task-local tree memory, persistent
cross-task experience.

This branch packages short-term and long-term memory without a Wiki. API credentials, model
weights, historical winning kernels, experimental RSI code and machine-specific
run artifacts are not included. The main search profile is `main_memory_longterm`.

## Requirements

- Python 3.10 for the pinned KernelBench integration.
- A PyTorch installation compatible with your GPU and driver for real benchmarks.
- A CUDA toolkit and C++ compiler for generated CUDA extensions.
- Access to a model/API of your choice, or separately downloaded local weights.

Real-device use currently targets CUDA. Backend-scoped experience does not
establish cross-architecture performance or validated NPU support.

## Installation

```bash
git clone --branch feature/memory-only --recurse-submodules https://github.com/yijiaduan111/KernelWeaver.git
cd KernelWeaver
python -m pip install -e ".[benchmark,test]"
git -C KernelBench apply --check ../patches/kernelbench-compat.patch
git -C KernelBench apply ../patches/kernelbench-compat.patch
cp .env.example .env
```

Install a suitable PyTorch build first if your driver requires a particular
version. Configure your own key, endpoint and model in `.env`. Never commit it.
The `all_api` route sends Plan, Code, Debug and Search roles through the selected
provider. It does not require local cudaLLM weights.

KernelBench is pinned to `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
The [compatibility patch](patches/README.md) preserves index/mask dtypes and
provides opt-in bounded-memory correctness checking without relaxed tolerances.
If downloading a source ZIP, clone KernelBench into `KernelBench/`, check out
that commit, and apply the same patch. A ZIP does not include Git submodules.

## CPU smoke test

No API key, GPU or model download is needed:

```bash
python stark_cli.py run-demo --task square_list --experiment quick   --provider mock --route-config all_mock --search-config main_memory_longterm   --max-attempts 2 --output-dir runs/demo_memory_only
```

## Generate and evaluate kernels

One task:

```bash
CUDA_VISIBLE_DEVICES=0 python stark_cli.py run-kernelbench   --experiment main --level 1 --problem-id 25 --backend cuda   --provider openai-compatible --route-config all_api   --search-config main_memory_longterm --output-dir runs/memory_only_single
```

The representative 15-task development set:

```bash
CUDA_VISIBLE_DEVICES=0 python stark_cli.py run-kernelbench-batch   --experiment main --task-config main_l1_15 --backend cuda   --provider openai-compatible --route-config all_api   --search-config main_memory_longterm --output-dir runs/memory_only_15
```

Select `--provider claude-compatible` for a native Claude-compatible endpoint,
`--provider micu-compatible` for the optional MicuAPI configuration, or
`--route-config codeagent_cudallm` for separately configured local code generation.
Set the model available to your own account through the corresponding environment
variables. Generated programs execute real code. See [SECURITY.md](SECURITY.md).

Each search uses five candidate attempts by default. Override with
`--max-attempts`. Start each independent experiment in a fresh output directory.
The batch reads/writes its own `long_term_memory.json` across ordered tasks.

For a detached run, use the same arguments with `scripts/run_batch.sh --detach`.
Store credentials in a private `.env` or use `--env-file`. Launcher scripts retain
non-secret settings only. `tmux` is required for detached launch.

## How memory is organized

| Component | Role |
| --- | --- |
| Short-term tree memory | Current/root/related programs, measured feedback and leaderboard inside one task |
| Long-term experience | Advisory successes and failures from earlier tasks in the same run |

The journal starts empty for a fresh output directory and is separate for each
independent run. Wiki corpus files, Wiki retrieval and Wiki prompt injection are
removed from this branch. Details and ablation profiles are in
[docs/memory.md](docs/memory.md).

## Validation

```bash
PYTHONPATH="$PWD/KernelBench/src" python -m unittest discover -s tests
python scripts/check_release.py
```

Tests use mocked providers or CPU fixtures. Real API/GPU experiments are separate.
This release makes no new speedup claim. Historical reports in
`docs/experiment_reports/` describe earlier configurations and should not be
attributed to this packaged profile. Reusing observed tasks or journal entries
is development-set adaptation, not an unseen-task generalization result.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```text
src/                 agents, tree/context, persistent experience and evaluators
configs/             API routes, search budgets, task and runtime settings
patches/             explicit compatibility patch for the pinned KernelBench
scripts/             launchers and public-checkout checks
tests/               CPU/unit regression checks
docs/                architecture, usage and historical reports
KernelBench/         upstream dependency as a pinned Git submodule
```

## License

Project code is MIT licensed. Third-party components retain their original
licenses. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
