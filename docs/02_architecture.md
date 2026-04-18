# Architecture

## Core Modules

- `src/core/bridge.py`: bridge from KernelBench tasks to the STARK-style workflow
- `src/core/workflow.py`: main workflow implementations
- `src/core/tree.py`: search-tree state
- `src/core/context.py`: role-specific context construction

## Agent Modules

- `src/agents/plan_agent.py`: planning step
- `src/agents/code_agent.py`: code generation step
- `src/agents/debug_agent.py`: debugging step

## Provider Modules

- `src/providers/openai_provider.py`: OpenAI-compatible API provider
- `src/providers/cudallm_provider.py`: local cudaLLM provider
- `src/providers/role_router.py`: per-role routing

## Evaluation Modules

- `src/evaluation/base.py`: shared evaluator helpers
- `src/evaluation/demo.py`: demo and Triton evaluators
- `src/evaluation/evaluator_paper.py`: official KernelBench-based evaluator
- `src/evaluation/validation.py`: saved-run revalidation
