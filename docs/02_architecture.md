# Architecture

## Core Modules

- `src/core/loader.py`: thin KernelBench loader that reads official problems and builds generic `ModelNew` scaffolds
- `src/core/workflow.py`: STARK-style workflow implementations
- `src/core/tree.py`: search-tree state and leaderboard tracking
- `src/core/context.py`: role-specific dynamic context construction

## Agent Modules

- `src/agents/plan_agent.py`: selects the optimization intent and grounded edit target
- `src/agents/code_agent.py`: implements the plan in the selected backend DSL
- `src/agents/debug_agent.py`: repairs compile, runtime, and correctness failures

## Provider Modules

- `src/providers/openai_provider.py`: OpenAI-compatible API provider
- `src/providers/claude_provider.py`: Claude-compatible API provider
- `src/providers/cudallm_provider.py`: local cudaLLM provider
- `src/providers/role_router.py`: per-role provider routing

## Evaluation Modules

- `src/evaluation/base.py`: shared evaluator helpers
- `src/evaluation/demo.py`: demo evaluator
- `src/evaluation/evaluator_paper.py`: official KernelBench-based evaluator
- `src/evaluation/validation.py`: saved-run revalidation

## Experiment Modules

- `src/experiment/batch_runner.py`: manifest loading and batch summaries
- `src/experiment/report_builder.py`: aggregate reports for completed runs

## Data Flow

1. `KernelBenchLoader` loads an official KernelBench problem into `TaskSpec`.
2. `workflow.py` initializes the root node from the generic `ModelNew` scaffold.
3. `PlanAgent` chooses the optimization strategy and structural anchor to edit.
4. `CodeAgent` emits a full updated Python module for the selected backend.
5. `KernelBenchPaperEvaluator` compiles, checks correctness, and measures runtime.
6. The search tree records candidates, failures, speedups, and the best node.
