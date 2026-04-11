# Architecture

## Core Flow
- `src/core/bridge.py`: loads official KernelBench tasks into internal `TaskSpec`
- `src/core/workflow.py`: STARK workflow and related workflow modes
- `src/core/tree.py`: search tree, leaderboard, pruning state
- `src/core/context.py`: dynamic context construction for each role

## Agent Layer
- `src/agents/plan_agent.py`
- `src/agents/code_agent.py`
- `src/agents/debug_agent.py`

These files are intentionally thin. They pass role-specific requests to providers.

## Provider Layer
- `src/providers/openai_provider.py`: OpenAI-compatible API backend
- `src/providers/cudallm_provider.py`: local full-weight cudaLLM backend
- `src/providers/mock_provider.py`: deterministic smoke-test backend
- `src/providers/role_router.py`: per-role provider dispatch

## Evaluation Layer
- `src/evaluation/evaluator_local.py`: local developer evaluator
- `src/evaluation/evaluator_paper.py`: paper-style KernelBench path
- `src/evaluation/validation.py`: replay and shadow validation

## Experiment Layer
- `src/experiment/batch_runner.py`: manifest loading and batch summaries
- `src/experiment/report_builder.py`: paper-style report generation
