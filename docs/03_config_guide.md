# Config Guide

## Main Idea
Most users should start with one experiment file under `configs/experiments/`.
An experiment file combines several lower-level config layers.

## Config Layers
- `experiments/`: full preset users select from CLI
- `tasks/`: which problems to run
- `models/providers/`: provider connection defaults
- `models/routes/`: plan/code/debug/search routing
- `search/`: search budget and temperatures
- `evaluation/evaluators/`: `local` or `paper`
- `evaluation/measurement/`: warmup, correctness trials, perf trials
- `runtime/`: paths, GPU visibility, env file

## Good Editing Order
1. Copy `configs/experiments/template_custom.yaml`
2. Change the referenced task, route, search, evaluator, or measurement config
3. Only edit the lower-level YAML files when you need a deeper change

## Common Questions
### Change the task list
Edit `configs/tasks/*.yaml`

### Change the base model or API endpoint
Edit `configs/models/providers/*.yaml`

### Make code role use a different model
Edit `configs/models/routes/*.yaml`

### Change search budget
Edit `configs/search/*.yaml`

### Change warmup and trial counts
Edit `configs/evaluation/measurement/*.yaml`
