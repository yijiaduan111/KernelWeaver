# Memory and Wiki configuration

The `memory_wiki_sm86` profile combines three separate channels.

| Channel | Lifetime | Contents | Writes |
| --- | --- | --- | --- |
| Short-term memory | One task search | Tree nodes, parent/current/root programs, related candidates, failures and leaderboard | Search loop |
| Long-term experience | One run directory | Compact measured successes and failures, implementation scope and advisory lessons | `long_term_memory.json` |
| Public Wiki | Frozen release | Operator contracts, general implementation guidance, architecture limits and source records | Explicit review and freeze only |

The journal starts empty in a fresh output directory. A batch shares it across
its ordered tasks. Reusing an output directory can reuse experience and must be
reported. Journal records do not become Wiki facts automatically. The current
journal is not protected against simultaneous independent writers, so parallel
experiments must use separate output directories.

Wiki retrieval identifies operator APIs and import aliases in the reference
program. It does not match problem IDs, task titles, fixed shapes or outcomes.
The checked-in corpus targets CUDA SM86. Unknown operators get a clearly marked
general fallback; another backend receives no CUDA pages. The corpus was
assembled after development-set inspection, so this release does not establish
unseen-task generalization or an unbiased performance gain.

Pages identify their sources and the corpus has a SHA256 lock. A valid lock
proves file consistency, not factual correctness or kernel speed. Source review
and actual candidate correctness/performance evaluation remain necessary.

## Profiles

| Search profile | Short-term context | Journal | Public Wiki |
| --- | --- | --- | --- |
| `main_memory_off` | Reduced to current/root context | Off | Off |
| `main_memory_on` | On | Off | Off |
| `main_memory_longterm` | On | On | Off |
| `main_memory_public_wiki_sm86` | On | Off | On |
| `memory_wiki_sm86` | On | On | On |

`main_memory_off` removes related-node and leaderboard context, but retains the
current program and root needed to generate a candidate. It is not a claim that
the entire optimizer is stateless. All five profiles use the same five-candidate
budget. Override it explicitly with `--max-attempts` when needed.

The public-only Wiki profile deliberately prohibits a cross-task journal.
The combined profile uses `operator_memory_public_corpus: true` with
`operator_memory_public_only: false`: the **Wiki** still comes only from the
frozen public corpus while empirical experience remains a separate channel.

## Inspecting and extending knowledge

```bash
python scripts/audit_public_wiki.py
python scripts/audit_public_wiki.py --manifest configs/tasks/main_l1_15.yaml \
  --report runs/wiki_coverage.json
```

The coverage command reads benchmark sources and retrieves pages without an API
request or GPU execution. Coverage is not correctness or performance.
After reviewing a deliberate edit, run `python scripts/audit_public_wiki.py --freeze`
and commit the content and lock together. Do not re-freeze merely to bypass an
unexpected integrity error.

This branch excludes task-mined Wiki pages, historical winning kernels and RSI
knowledge/policy evolution. Optional profiling hooks remain disabled in the
combined profile. No new GPU benchmark result is claimed by the release cleanup.
