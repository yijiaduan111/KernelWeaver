# Quick Start

## 1. Pick an experiment
The main presets are:
- `paper_mini` (recommended default baseline)
- `quick_local` (fast smoke preset)
- `paper_full`

## 2. Run a small batch
```bash
python stark_cli.py run-kernelbench-batch   --experiment paper_mini   --output-dir runs/paper_mini
```

## 3. Run one KernelBench task
```bash
python stark_cli.py run-kernelbench   --experiment paper_mini   --level 1   --problem-id 25   --output-dir runs/l1_p25
```

## 4. Run one built-in demo
```bash
python stark_cli.py run-demo   --task square_list   --experiment quick_local   --output-dir runs/demo_square_list
```

## 5. Inspect one saved run
```bash
python stark_cli.py show-run runs/l1_p25/run.json
```
