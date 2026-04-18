# Config Guide

## The Main Idea

Every experiment is mainly described by four things:
- `tasks`
- `backend`
- `route`
- `profile`

The `profile` is only a short name for:
- `search`
- `evaluator`
- `measurement`

## Config Folders

- `configs/experiments/`: high-level presets
- `configs/tasks/`: which problems to run
- `configs/models/providers/`: provider defaults
- `configs/models/routes/`: which provider each role uses
- `configs/search/`: search budget and temperatures
- `configs/evaluation/evaluators/`: evaluator path and reference modes
- `configs/evaluation/measurement/`: correctness and timing counts
- `configs/runtime/`: paths and device settings

## Which File To Edit

- Change the task list: edit `configs/tasks/`
- Change model routing: edit `configs/models/routes/`
- Change evaluator reference modes: edit `configs/evaluation/evaluators/`
- Change correctness/timing budget: edit `configs/evaluation/measurement/`
- Change the recommended preset combination: edit `configs/experiments/`
