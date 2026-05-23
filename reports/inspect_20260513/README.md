# KernelWeaver Inspection Pack

This pack contains read-only diagnostics for recent semantic + deliberation experiments.

Recommended reading order:
1. `cudallm_15_summary.md`
2. `cudallm_15_P25.md` for a positive case
3. `cudallm_15_P42.md` for a pooling/reduction positive-but-risky case
4. `cudallm_15_P1.md` and `cudallm_p1_single_P1.md` for batch-vs-single comparison
5. `claude_15_summary.md`, then `claude_15_P20.md` and `claude_15_P25.md`

Raw files are under `raw/`:
- `summary.json`: aggregate experiment table
- `launcher.log`: runtime/evaluator log
- `*/run.json`: semantic profile, strategy portfolio, search nodes
- `*/best_code.py`: best selected candidate code
