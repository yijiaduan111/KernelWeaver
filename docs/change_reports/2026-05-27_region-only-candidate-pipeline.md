# Region-Only Candidate Pipeline Report

Date: 2026-05-27  
Project: KernelWeaver  
Branch: `feature/model-deliberation-v0`  
Base commit: `cc17343 Protect CUDA scaffold helpers`  
Main run(s): temporary smoke runs under `runs/.tmp_region_only_smoke_*`?????????  
Baseline run(s): previous Region Scaffold V2 / Backend Contract Layer runs on the same branch

## 1. Summary

- Change result: **Partially successful**.
- One-sentence conclusion: ???????????????? `region_patches`????? `anchor_patches`/full-module fallback ? scaffold ???????????? 15 ????????
- Engineering result: P42 ? 5-attempt smoke ?????`compile_rate=1.0`?`correct_rate=1.0`?best speedup `1.686x`????? anchor patch ???
- Main caveat: P25 smoke ?????? 30 ?????? CUDA `nvcc/cicc` ??????? CUDA extension ???????????????

## 2. Why We Changed This

??? Region Scaffold V2 ??? CUDA scaffold helper?extension loader?`user_helpers` ?????????????????????????

- ????`region_patches`???? LLM ???? editable region ? body?
- ????`anchor_patches` / `anchor_name` alias / full-module output fallback????????? region scaffold?

?????????

1. **?????**????????????????? normalize ????????????????
2. **??????**??????? anchor patch?????? parser??????????????? workflow/debug ?????????

???????????? speedup??????????????? KernelBench candidate ?????? region patch + hygiene + static/contract check ???

## 3. What Changed

### 3.1 Candidate normalization ?? region-only

?????`src/core/candidate.py`

- `normalize_candidate()` ????? `region_patches` JSON?
- `anchor_patches`??? `anchor_name`?`patches`?`edits` ???? invalid patch payload?
- `region_patches` ??????? `region` ???`anchor_name` alias ??????
- ?? editable region ? parent scaffold?full-module output ???? fallback ??????? `full_module_region_task`?
- ? region scaffold ? full Python module fallback ??????????? KernelBench/? scaffold ???

### 3.2 cudaLLM parser ??? region patches

?????`src/providers/cudallm_provider.py`

- `_normalize_patch_response()` ??? `region_patches`?
- ???? `anchor_patches` / `patches` / `edits` / single `anchor_name`?
- ??? `anchor_name` ????? `region`?
- ??? local cudaLLM ???????? provider ??????????

### 3.3 ??? anchor edit ??

?????`src/utils.py`

- ?? `apply_anchor_edit()`?
- ?? `replace_anchor_body()`?
- ?? `extract_anchor_names()` ? `preserve_anchor_scaffold()`???????????? IMPROVE marker/anchor marker ??? editable regions???? marker discovery???? patch ?????

### 3.4 Hygiene ?? `user_helpers`

?????`src/core/hygiene.py`

- `user_helpers` ?? Python region hygiene ???
- ????? helper ??? `forward_stmt_*`?`init_body`?`forward_body` ????? Python ??/???????

### 3.5 Tests ??????

?????

- `tests/test_candidate_guards.py`
- `tests/test_kernelbench_flow.py`

?????

- ?? ?anchor patch ??? apply? ????? ?anchor patch ???????
- ??/?? region patch ???????
- ?? full-module output ? region task ????????
- `test_kernelbench_flow.py` ???? `apply_anchor_edit()`??? `apply_region_patches()`?

## 4. Validation Setup

### Static / unit-level checks

- Branch: `feature/model-deliberation-v0`
- Environment: server `/data/dyj/KernelWeaver`?conda env `stark`
- Commands:
  - `python -m py_compile src/utils.py src/core/candidate.py src/core/hygiene.py src/core/regions.py src/providers/cudallm_provider.py tests/test_candidate_guards.py tests/test_kernelbench_flow.py`
  - Direct invocation of pytest-style test functions because server env has no `pytest` package.
  - `git diff --check`

Result:

- `36` direct test functions passed.
- `py_compile` passed.
- `git diff --check` passed.

### Smoke 1: P25

- Task: L1 P25 Swish
- Backend: CUDA
- Route: `codeagent_claude`
- Profile: `main`
- Attempts: `5`
- GPU setup: `CUDA_VISIBLE_DEVICES=0,1`?`CUDA_HOME=/usr/local/cuda-12.8`

Result:

- Run was stopped and cleaned after more than 30 minutes.
- It did not produce `run.json` before cleanup.
- Process was active in evaluator worker and then `nvcc/cicc` compilation; no evidence of API hang or region parser failure.
- This smoke exposed a CUDA compilation-cost issue rather than a region-only protocol issue.

### Smoke 2: P42

- Task: L1 P42 MaxPool2d
- Backend: CUDA
- Route: `codeagent_claude`
- Profile: `main`
- Attempts: `5`
- GPU setup: `CUDA_VISIBLE_DEVICES=0,1`?`CUDA_HOME=/usr/local/cuda-12.8`

Result:

- Status: `ok`
- Strategy count: `10`
- Best node: non-root candidate `n2`
- Candidate total: `5`
- Compile count: `5`
- Correct count: `5`
- Best speedup: `1.6862170087976538x`
- Temporary run directory was cleaned after extracting results.

## 5. Macro Results

Because this was a targeted smoke rather than full `main_l1_15` experiment, macro metrics only apply to P42.

| Metric | Before | After | Interpretation |
|---|---:|---:|---|
| Compile Rate | n/a | `1.0` | P42 5 candidates all compiled successfully. |
| Correct Rate | n/a | `1.0` | P42 5 candidates all passed correctness. |
| Success@5 | n/a | `1.0` | P42 solved. |
| Fast@5 | n/a | `1.0` | Best candidate faster than torch eager. |
| Best Speedup | n/a | `1.686x` | Region-only path did not block optimization. |
| Geomean Speedup | n/a | `1.686x` | Single-task smoke only. |
| Solved Tasks | n/a | `1/1` | P42 solved. |
| Fast Tasks | n/a | `1/1` | P42 improved over reference. |

## 6. Per-Task Results

| Task | Before Best | After Best | Compile | Correct | Main Failure / Note |
|---|---:|---:|---:|---:|---|
| L1 P42 MaxPool2d | n/a | `1.686x` | `5/5` | `5/5` | Smoke passed; no protocol/scaffold failure. |
| L1 P25 Swish | n/a | incomplete | incomplete | incomplete | Stopped after long `nvcc/cicc` compilation; no `run.json`. |

## 7. Case Studies

### P42 MaxPool2d: desired behavior

- What happened: P42 completed with all 5 candidates compiling and passing correctness.
- Evidence: summary showed `candidate_total_count=5`?`candidate_compile_count=5`?`candidate_correct_count=5`?best speedup `1.686x`?
- Interpretation: Region-only candidate protocol did not damage normal candidate generation/evaluation. It also did not introduce visible scaffold corruption.

### P25 Swish: CUDA compile bottleneck

- What happened: P25 entered evaluator worker and repeatedly reached `nvcc/cicc` compilation, but did not produce a final run within the observed time.
- Evidence: process tree showed `python -m src.evaluation.worker` and child `nvcc` / `cicc` under `/usr/local/cuda-12.8/bin/nvcc` compiling generated Torch extension sources.
- Interpretation: this is not a model API or patch parser failure. It is the cost of compiling distinct `load_inline` CUDA extensions for multiple generated candidates.

## 8. Failure Analysis

### Code protocol / scaffold issues

- Old protocol is now rejected rather than silently adapted.
- No `applied_anchor_patch` appeared in smoke outputs.
- No `full_module_region_task` appeared in the successful P42 run.
- No `python_region_syntax_error` appeared in P42.

### CUDA runtime / memory safety

- P42 did not expose runtime memory safety problems.
- P25 was stopped during compilation, so runtime behavior was not evaluated.

### Correctness / semantic mismatch

- P42 had `5/5` correctness, so no semantic mismatch was observed there.
- P25 has no correctness conclusion because it did not finish.

### API / infrastructure

- P25 and P42 both progressed beyond launch into evaluator/compile stage, so this validation did not point to API connectivity as the main issue.

### Resource / compile cost

- The main new observation is high CUDA extension compile latency.
- Root cause is likely repeated `torch.utils.cpp_extension.load_inline` compilation with unique hash-based extension names per candidate.
- Even simple tasks can become slow if generated CUDA code triggers expensive template/header compilation through PyTorch extension infrastructure.

## 9. Did This Meet the Goal?

Original goal: remove legacy `anchor_patches` compatibility, keep only clean `region_patches`, and ensure the resulting candidate pipeline remains runnable.

| Acceptance criterion | Result | Evidence |
|---|---|---|
| Legacy `anchor_patches` no longer accepted | Met | Candidate normalize tests reject `anchor_patches`; parser no longer reads it. |
| `anchor_name` alias removed | Met | Region patch entries require `region`; alias is rejected. |
| Full-module bypass blocked for region scaffold | Met | `full_module_region_task` added for region-scaffold tasks. |
| Old edit helper removed | Met | `apply_anchor_edit` / `replace_anchor_body` deleted from `src/utils.py`; tests use `apply_region_patches`. |
| Static tests pass | Met | 36 direct tests passed; compile check passed. |
| Smoke run still works | Met for P42 | P42 completed with 5/5 compile and correctness. |
| Full 15-task performance validated | Not yet | Only targeted smoke was run. |
| CUDA compile cost solved | Not a target / Not solved | P25 showed long `nvcc/cicc` compile time. |

Final judgment: **Partially successful**. Engineering cleanliness and protocol unification were achieved. Performance impact and full-task robustness still need a full `main_l1_15` experiment or at least a broader smoke set.

## 10. Next Steps

- Immediate action: commit this cleanup as a stable engineering checkpoint, because it makes the candidate pipeline cleaner and easier to debug.
- Next experiment: run `main_l1_15 + CUDA + chosen code agent + main` after confirming GPU availability, then compare compile/correct/failure distributions with the previous Region Scaffold V2 runs.
- Follow-up engineering topic: separately plan CUDA compile-cost control. Possible directions include compile timeout visibility, pre-filtering expensive candidates, better extension cache reuse, and clearer logging of candidate compile durations.
- Longer-term direction: keep region-only as the base protocol for future semantic/deliberation/backend contract work; avoid reintroducing anchor-patch compatibility unless a concrete external API requires it.
