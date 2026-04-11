# Results Guide

## Single Task Output
A single task directory usually contains:
- `run.json`: full structured run record
- `best_code.py`: current best candidate code
- `validation.json`: replay validation result

## Batch Output
A batch output directory usually contains:
- `summary.json`: machine-readable batch summary
- `summary.csv`: spreadsheet-friendly summary
- one subdirectory per task

## Important Fields
- `run_profile`: the selected experiment
- `search_profile`: search config name
- `evaluator_profile`: evaluator config name
- `measurement_profile`: measurement config name
- `kernelbench_evaluator`: resolved evaluator kind, such as `local` or `paper`

## Useful CLI
```bash
python stark_cli.py show-run runs/some_task/run.json
python stark_cli.py verify-kernelbench runs/some_task/run.json
python stark_cli.py report-paper runs/paper_mini --output-dir runs/paper_report
```
