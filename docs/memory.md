# Short-term and long-term memory

This branch contains two memory channels and no Wiki loader or corpus.

| Channel | Lifetime | Contents |
| --- | --- | --- |
| Short-term tree context | One task search | Current/root/related programs, failures, measured runtimes and leaderboard |
| Long-term experience journal | One output directory | Compact advisory successes and failures from completed task searches |

`main_memory_longterm` is selected by `--experiment main`. Its defaults are five
candidate attempts, `context_limit: 5`, journal retrieval `top_k: 4`, and no
extra exploration reservation. Adjust the budget explicitly with `--max-attempts`.

The journal is `long_term_memory.json` at the output root. A fresh directory
starts empty. In a batch, completed tasks append records and later tasks can
retrieve them. Entries include the operator type, backend, strategy, outcome,
failure information and advisory lesson. Recorded speedups are observations,
not guarantees. Incorrect or noisy interpretations can still enter a journal,
so experiments must track its initial state and task order.

Use a different output directory for each independent run. Concurrent writers
must not share one journal. Reusing a directory may reuse prior experience and
must not be reported as a cold-start result. No historical answers or prefilled
experience are shipped in this branch.

## Ablation profiles

| Profile | Related-node/leader context | Persistent journal |
| --- | --- | --- |
| `main_memory_off` | Disabled | Disabled |
| `main_memory_on` | Enabled | Disabled |
| `main_memory_longterm` | Enabled | Enabled |

All three profiles use the same five-candidate budget. The current/root programs
remain available even in `main_memory_off`, so it is not a fully stateless
optimizer. Each candidate still goes through the configured correctness and
timing evaluator. No root-fallback statistic should be confused with a generated
candidate's success or speedup.

## Scope and compatibility

Backend-specific entries are not retrieved across mismatched backends. Current
device experiments target CUDA; this does not validate portability to NPU or
guarantee transfer across GPU architectures or shapes.

Wiki files, retrieval and prompt injection are absent. A few legacy dataclass
and report fields remain to read previously saved artifacts. They default to
empty values. Attempts to enable removed Wiki settings fail explicitly before
evaluation. Use `feature/memory-wiki` if that separate capability is required.

This branch does not include RSI updates, automatic knowledge promotion, learned
search policies or historical benchmark winners. Packaging and CPU tests do not
establish GPU performance improvements.
