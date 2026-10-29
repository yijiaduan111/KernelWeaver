# Contributing

Use a feature branch and describe the problem, resulting behavior and validation
in your pull request. Keep functional changes separate from benchmark results.

```bash
python -m pip install -e ".[benchmark,test]"
git submodule update --init --recursive
git -C KernelBench apply ../patches/kernelbench-compat.patch
PYTHONPATH="$PWD/KernelBench/src" python -m unittest discover -s tests
python scripts/check_release.py
```

Tests use mocks or CPU fixtures and do not require API keys or a GPU. CUDA
performance experiments are separate and must declare the model, candidate
budget, hardware, task order, initial journal state, failures and total costs.

This branch contains task-local tree context and a per-run empirical journal.
Use a fresh output directory for each independent experiment. Do not add historical
winning kernels or automatically promote noisy observations into universal rules.
Wiki and RSI features belong in separate branches.

Never commit `.env` variants, access tokens, model weights, run directories or
generated kernels. Contributions are provided under the project's MIT license,
with third-party notices retained where applicable.
