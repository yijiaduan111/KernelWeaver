# Config Layout

Configs are split by responsibility so new users do not need to edit one large file.

## Top Level
- `experiments/`: the main presets users should select first
- `tasks/`: task manifests
- `models/providers/`: provider connection defaults
- `models/routes/`: which provider each role uses
- `search/`: search tree and attempt budget
- `evaluation/evaluators/`: evaluator type such as `local` or `paper`
- `evaluation/measurement/`: warmup, trials, timing settings
- `runtime/`: paths, GPU visibility, env file

## Recommended Usage
Start from `configs/experiments/*.yaml`.
Only go deeper when you need to customize one specific layer.
