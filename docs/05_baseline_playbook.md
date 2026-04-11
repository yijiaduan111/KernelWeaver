# Baseline Playbook

## Repo Roles
- `/data/dyj/STARK`: frozen reference repo
- `/data/dyj/KernelWeaver`: clean collaboration baseline

## Recommended Team Practice
1. Start with `paper_mini` for the default 9-task CUDA baseline
2. Use `quick_local` only when you need a very fast smoke check
3. Use `paper_full` only when you want the heavier paper-style setting

## Keep The Repo Clean
- Do not keep ad-hoc scripts in the repo root
- Do not commit `runs/`
- Do not leave temporary configs after debugging
- Prefer changing YAML configs instead of patching CLI for one-off runs

## When To Edit What
- Need a new task pack: add a file under `configs/tasks/`
- Need a new provider: add one provider config and, if needed, one route config
- Need a new experiment preset: add one file under `configs/experiments/`
