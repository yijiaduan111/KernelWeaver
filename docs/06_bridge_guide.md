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

In the current KernelWeaver baseline, the bridge is **curated-first**.
That means `load_official_problem(...)` only supports tasks that are explicitly enabled in:
- `src/core/bridge.py`
- `_SELECTED_TARGETS`

So the current baseline does **not** provide a fully open-ended auto-bridge fallback for every KernelBench task.
If a task is not enabled in `_SELECTED_TARGETS`, the bridge will reject it.

## Why this restriction exists

A STARK-style workflow usually needs more than the raw official file:
- a `ModelNew` scaffold
- grounded anchor regions
- stable candidate/reference entrypoints
- task-specific strategy hints in some cases
- backend-specific starter code in some CUDA cases

For the current baseline, we keep the supported set explicit so that the scaffold quality stays predictable.

## Where to add a new task

If you want to support a new KernelBench task, the main file is:
- `src/core/bridge.py`

The common places to edit are:

### 1. `_SELECTED_TARGETS`

Add one entry for `(level, problem_id)`.
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

If you want to run the task in a standard batch, add it to a manifest under:
- `configs/tasks/`

## Practical rule

Use the current bridge like this:
- supported selected task -> directly runnable
- unsupported new task -> add a curated bridge entry first

That is the accurate behavior of the current codebase.
