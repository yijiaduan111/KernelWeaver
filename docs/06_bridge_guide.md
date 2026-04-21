# 06 Bridge Guide

## What the bridge does

KernelBench and KernelWeaver do not use the same task format.
A bridge is needed because KernelWeaver does not optimize raw benchmark files directly.
It optimizes an internal `TaskSpec` object with extra structure.

In practice, the bridge converts one official KernelBench problem into a task object that contains:
- editable candidate source
- official reference source
- entrypoint names
- grounded edit anchors
- task metadata such as level, problem id, backend, and tags

Without this layer, the workflow would only see a plain benchmark file.
It would not know where local edits are allowed or how to build a stable scaffold.

## Current status in this baseline

In the current KernelWeaver baseline, the bridge is **curated-first with auto fallback**.

That means:
- curated tasks in `src/core/bridge.py` -> use hand-tuned metadata, shapes, and strategy hints
- unsupported official tasks -> fall back to a generic auto bridge

The auto bridge does these things automatically:
- load the official `Model`, `get_inputs`, and `get_init_inputs`
- build a grounded `ModelNew` scaffold
- generate reduced local test and benchmark cases
- support both `triton` and generic `cuda` scaffolds

So a task that is not listed in `_SELECTED_TARGETS` is no longer rejected by default.
It can still run through the official bridge path, but the scaffold quality is more generic.

## Why curated entries still matter

A STARK-style workflow usually needs more than the raw official file:
- a `ModelNew` scaffold
- grounded anchor regions
- stable candidate/reference entrypoints
- task-specific strategy hints in some cases
- backend-specific starter code in some CUDA cases

The auto bridge is enough to get a new task running.
But curated entries are still better when you want:
- task-specific reduced shapes
- stronger strategy hints
- better CUDA starter code
- more stable low-cost experiments

The current `configs/tasks/main_l1_15.yaml` set is kept as curated-first on purpose.

## When auto bridge is enough

Auto bridge is usually enough when you want to:
- quickly run a new official KernelBench task
- test a new task on the paper evaluator path
- build a first-pass Triton or generic CUDA scaffold

This is the normal path for:
- trying a new problem once
- extending a task list
- smoke testing a new backend route

## When manual enhancement is worth doing

If you want stronger results on a task, the main file is still:
- `src/core/bridge.py`

The common places to edit are:

### 1. `_SELECTED_TARGETS`

Add one curated entry for `(level, problem_id)`.
This is where you define:
- task name
- title
- tags
- strategy references
- other task metadata used by the scaffold

### 2. Backend scaffold helpers

If the generic scaffold is not enough, check backend-specific helper logic in:
- `src/core/bridge.py`

Typical reasons to edit it:
- better CUDA starter code
- task-specific exported bindings
- task-specific forward structure

### 3. Task manifests

If you want to run the task in a standard batch, add it to a task file under:
- `configs/tasks/`

## Practical rule

Use the current bridge like this:
- curated selected task -> best quality, directly runnable
- unsupported new task -> auto bridge can run it immediately
- unstable or expensive new task -> add a curated bridge entry and tune its reduced cases

That is the current behavior of the codebase.
