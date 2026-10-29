# Changelog

## 0.2.0 memory-only variant (unreleased)

- Default to `main_memory_longterm`: task-local tree context and a fresh per-run
  cross-task experience journal, with five candidate attempts by default.
- Remove the Wiki corpus, loader, retrieval profiles, prompt injection and audit
  script. Old Wiki opt-ins fail explicitly rather than silently changing behavior.
- Preserve MIT licensing, third-party notices, portable API routing, the pinned
  KernelBench dependency/compatibility patch and CPU validation.
- Keep model weights, credentials, generated outputs, historical winning kernels
  and experimental RSI features outside the release tree.

This parallel branch does not replace `feature/memory-wiki` or claim new GPU
benchmark results. Existing Git history is retained, so previously exposed
credentials still require independent revocation.
