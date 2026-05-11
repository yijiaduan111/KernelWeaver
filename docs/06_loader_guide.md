# 06 Loader Guide

## Why the loader exists

KernelWeaver runs STARK-style workflows on official KernelBench problems. The workflow still needs an internal `TaskSpec`, but task ingestion should stay thin and generic.

The loader therefore only does the minimum required work:
- locate the official KernelBench problem file
- read the original source code
- verify `Model`, `get_inputs`, and `get_init_inputs` exist
- build a generic `ModelNew` scaffold
- add structural anchors such as `helpers`, `init_body`, and `forward_body`
- record metadata such as level, problem id, backend, and source path

## What the loader does not do

The loader does not contain per-task adapter logic:
- no curated task table
- no handwritten strategy catalog
- no `good_body`, `broken_body`, or `debug_body`
- no task-specific CUDA starter kernel
- no manual selection of the best edit region

This keeps the baseline close to STARK: the `PlanAgent` is responsible for selecting the edit target from the structural anchors during the workflow.

## Multi-DSL behavior

The same official KernelBench problem enters through the same loader. Backend differences are represented as generic scaffold sections:
- `cuda`: `helpers`, `cuda_cpp`, `cuda_cu`, `init_body`, `forward_body`
- `triton`: `helpers`, `init_body`, `forward_body`
- `tilelang`: `helpers`, `tilelang_kernel`, `init_body`, `forward_body`
- `cute`: `helpers`, `cute_kernel`, `init_body`, `forward_body`

The backend-specific provider prompts explain how each DSL should use those sections.

## Practical rule

For new KernelBench tasks, do not add per-task adapter code. Add the task to a manifest under `configs/tasks/`, choose a backend and route, then let the loader and agents handle the workflow.
