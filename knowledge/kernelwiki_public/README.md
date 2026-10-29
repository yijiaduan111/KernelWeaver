# Frozen public-source Wiki

This corpus contains project-authored summaries of primary documentation and
papers, general operator contracts, architecture constraints and explicit
fallback guidance. It contains no historical winning kernels or benchmark
performance results. Source links and provenance are recorded in frontmatter.

- `sources/`: primary-source records and summaries.
- `wiki/operators/`: operator contracts and conditional guidance.
- `wiki/hardware/`: architecture-specific constraints (currently CUDA SM86).
- `wiki/contracts/`: explicit fallback for unsupported operator families.
- `data/`: operator API and alias mapping.
- `corpus.lock.json`: SHA256 lock for runtime content and retrieval data.

Retrieval uses API structure rather than task identifiers, fixed shapes or
outcomes. A `reviewed` source label does not certify a generated kernel or a
performance gain. This corpus was assembled after development-set inspection,
so a new frozen holdout is needed for prospective generalization claims.

`memory_wiki_sm86` combines this read-only corpus with a separate per-run
experience journal. `main_memory_public_wiki_sm86` disables that journal for
public-only knowledge experiments. Neither profile automatically updates pages.
Task-mined knowledge and RSI drafts are not distributed in this branch.

From the repository root:

```bash
python scripts/audit_public_wiki.py
python scripts/audit_public_wiki.py --manifest configs/tasks/main_l1_15.yaml \
  --report runs/wiki_coverage.json
```

Coverage is retrieval coverage only, not correctness or speed. Review all
runtime content before intentionally updating the lock with `--freeze`.
See [memory documentation](../../docs/memory_wiki.md) for the channel boundaries.
