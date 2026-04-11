# 06 Bridge Guide

## What the bridge does

KernelBench and KernelWeaver do not use the same task format.
A bridge is needed because KernelWeaver does not optimize raw benchmark files directly.
It optimizes an internal task object with extra structure.

In practice, the bridge converts one official KernelBench problem into a `TaskSpec` that contains:
- the editable candidate source
- the official reference source
- the entrypoint names
- grounded edit anchors
- task metadata such as level, problem id, backend, and tags
- optional local test cases and local benchmark cases

Without this bridge layer, the STARK-style workflow would only see a plain Python file.
It would not know where local edits are allowed, how to build a stable scaffold, or how to feed the task into evaluators in a consistent way.

## Why KernelBench needs bridging

A raw official KernelBench problem usually contains:
- `Model`
- `get_inputs()`
- `get_init_inputs()`

That is enough for the official benchmark.
It is not enough for a STARK-style search workflow.

KernelWeaver needs extra structure that does not exist in the raw file:
- `ModelNew` as the editable candidate class
- `# <<<IMPROVE:...>>>` anchor regions for constrained local edits
- a unified internal schema (`TaskSpec`)
- optional reduced local cases for fast iteration

So the bridge is not optional glue.

## Current bridge modes

KernelWeaver currently supports two bridge modes.

### 1. Curated override

Curated tasks use hand-written bridge metadata.
They usually have:
- reduced local test cases
- reduced local benchmark cases
- cleaner tags and names
- optional strategy hints
- optional CUDA-specific starter scaffold improvements

These curated overrides live in:
- `src/core/bridge.py`

The main curated table is:
- `_SELECTED_TARGETS`

### 2. Auto bridge

If a task is not listed in the curated override table, KernelWeaver now falls back to auto bridge.

Auto bridge does the following automatically:
- locate the official problem file
- parse `Model`, `get_inputs()`, and `get_init_inputs()`
- build a `ModelNew` scaffold
- insert grounded edit anchors
- return a valid internal `TaskSpec`

This means a new official KernelBench problem can now be loaded without writing a manual bridge first.

## What auto bridge supports today

Auto bridge supports the following well:
- official KernelBench tasks with the standard file structure
- STARK-style task loading into `TaskSpec`
- grounded edit anchors for local code editing
- `triton` scaffold generation
- generic `cuda` scaffold generation
- the official-style `paper` evaluator path

In other words, if you want to run a new official KernelBench task with the paper evaluator, auto bridge is usually enough.

## What auto bridge does not try to solve fully

Auto bridge does not try to guess a perfect reduced local testing profile for every task.
That is the main reason manual enhancement still exists.

For a new task, auto bridge does **not** automatically provide:
- reduced local `test_cases`
- reduced local `benchmark_cases`
- task-specific strategy hints
- task-specific CUDA starter kernels
- carefully tuned local shapes for fast smoke testing

## Why `paper` evaluator works for auto-bridged tasks

The `paper` evaluator uses the official KernelBench evaluation path.
It does not depend on our own reduced local cases.

That is why auto-bridged tasks are mainly intended for:
- `workflow=stark`
- `evaluator=paper`
- official-style evaluation

This is the lowest-friction path for a new task.

## Why local evaluator needs manual enhancement

The local evaluators need explicit local cases from the bridge layer.
They directly consume:
- `task.test_cases`
- `task.benchmark_cases`

These local cases are used for:
- correctness checks
- runtime measurement
- fast small-scale debugging

For many tasks, these cases are task-specific.
A safe reduced shape for one task may be wrong, misleading, too slow, or even invalid for another task.

That is why KernelWeaver does not auto-generate aggressive local reduced cases for every task.
It is safer to require manual enhancement when you want strong local iteration support.

## Why CUDA often needs more manual work

For `cuda`, auto bridge can build a generic scaffold.
That is enough to give the model a valid place to write:
- helper code
- C++ binding code
- CUDA kernel code
- Python forward logic

However, generic CUDA scaffolds are only a starting point.
For better local performance and debugging, some tasks benefit from manual enhancement such as:
- custom CUDA starter kernels
- task-specific extension exports
- more helpful CUDA-specific strategy hints

So the rule is:
- auto bridge is enough to start
- manual CUDA enhancement improves usability and quality

## When you should use auto bridge only

Use auto bridge only when:
- the task is a new official KernelBench problem
- you want to test the official-style path quickly
- you plan to use the `paper` evaluator
- you do not need reduced local smoke cases yet

Recommended command:

```bash
conda run -n stark python stark_cli.py run-kernelbench \
  --experiment paper_mini \
  --kernelbench-root /path/to/KernelBench \
  --level 2 \
  --problem-id 43 \
  --output-dir runs/l2_p43
```

## When you should add manual enhancement

In short:
- official benchmark path -> auto bridge is usually enough
- local development path -> manual enhancement is often worth it

## What to edit for manual enhancement

If you want to manually enhance one task, the main file is:
- `src/core/bridge.py`

The usual places to edit are:

### 1. `_SELECTED_TARGETS`

Add one entry for `(level, problem_id)`.
This is where you define:
- task name
- title
- tags
- init args / init kwargs
- reduced local test shapes
- reduced local benchmark shapes
- input kind
- optional strategy specs

### 2. `_strategy_catalog_for_backend(...)`

Edit this only if the task needs backend-specific strategy hints.
Typical cases:
- custom CUDA forward call hints
- task-specific anchor instructions
- safer debug repair hints

### 3. `_cuda_backend_bodies(...)`

Edit this only if the generic CUDA scaffold is not enough.
Typical cases:
- a better starter extension layout
- task-specific C++ export code
- task-specific CUDA kernel starter code

### 4. task manifests under `configs/tasks/`

If you want to run the task in a standard batch, add it to the task manifest you care about.
Examples:
- `configs/tasks/kb9_cuda.yaml`
- `configs/tasks/kb9_triton.yaml`
- `configs/tasks/kb15_representative.yaml`

## Practical workflow for a new task

### Option A: quick official-style run

1. Do not write a manual bridge
2. Use auto bridge
3. Run with `paper_mini` or `paper_full`
4. Check whether the task loads and runs correctly

### Option B: promote one task into a curated task

1. First run it with auto bridge
2. If the task is important, add a curated override
3. Add reduced local cases
4. Add strategy hints if needed
5. Add CUDA scaffold improvements if needed
6. Add it to your task manifest

## Support summary

### Auto bridge
- official KernelBench structure: yes
- STARK task loading: yes
- grounded anchors: yes
- triton scaffold: yes
- generic cuda scaffold: yes
- paper evaluator: yes
- reduced local cases: no
- task-specific strategy hints: no
- task-specific CUDA starter code: no

### Curated override
- official KernelBench structure: yes
- STARK task loading: yes
- grounded anchors: yes
- triton scaffold: yes
- cuda scaffold: yes
- paper evaluator: yes
- reduced local cases: yes
- task-specific strategy hints: yes
- task-specific CUDA starter code: yes

## Bottom line

You do not need to hand-write a bridge for every new KernelBench task anymore.
Start with auto bridge.
Only add manual enhancement when you need better local debugging, better local evaluation, or better backend-specific starter code.
