"""Helpers for task manifests and batch summaries."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import yaml


def load_task_manifest(path: str | Path, kernelbench_root: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f'Could not find manifest: {manifest_path}')
    if manifest_path.suffix.lower() == '.json':
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        payload = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
    tasks = payload.get('tasks')
    if isinstance(tasks, list) and tasks:
        return {
            'name': payload.get('name', manifest_path.stem),
            'tasks': [_normalize_task(item, payload.get('backend')) for item in tasks],
        }
    source = str(payload.get('source') or '').strip()
    if source == 'kernelbench_all':
        if kernelbench_root is None:
            raise ValueError('kernelbench_root is required for source=kernelbench_all')
        backend = str(payload.get('backend') or 'triton')
        return {
            'name': payload.get('name', manifest_path.stem),
            'tasks': _discover_all_official_tasks(Path(kernelbench_root), backend=backend),
        }
    raise ValueError(f'Manifest must define a non-empty tasks list or source=kernelbench_all: {manifest_path}')


def batch_output_dir_name(alias: str, level: int, problem_id: int) -> str:
    safe_alias = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in alias)
    return f'{safe_alias}_l{level}_p{problem_id}'


def aggregate_batch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful = [row for row in rows if row['status'] == 'ok' and row.get('best_correct')]
    speedups = [float(row['speedup']) for row in successful if isinstance(row.get('speedup'), (int, float))]
    speed_metric_values = [
        float(row['speedup']) if row.get('best_correct') and isinstance(row.get('speedup'), (int, float)) else 0.0
        for row in rows
    ]
    failure_stage_counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get('failure_stage') or 'unknown')
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
    generated_total = sum(int(row.get('candidate_total_count') or 0) for row in rows)
    generated_compile = sum(int(row.get('candidate_compile_count') or 0) for row in rows)
    generated_correct = sum(int(row.get('candidate_correct_count') or 0) for row in rows)
    return {
        'task_count': total,
        'success_count': len(successful),
        'root_correct_rate': _ratio(sum(1 for row in successful if row.get('root_correct')), total),
        'non_root_correct_candidate_rate': _ratio(sum(1 for row in successful if row.get('non_root_correct')), total),
        'improved_over_reference_rate': _ratio(sum(1 for row in successful if row.get('improved_over_reference')), total),
        'best_node_is_root_rate': _ratio(sum(1 for row in successful if row.get('best_node_is_root')), total),
        'compile_rate': _ratio(generated_compile, generated_total),
        'correct_rate': _ratio(generated_correct, generated_total),
        'generated_candidate_count': generated_total,
        'median_speedup': statistics.median(speedups) if speedups else None,
        'best_speedup': max(speedups) if speedups else None,
        'failure_stage_distribution': failure_stage_counts,
        'paper_metrics': {
            'Success': _ratio(len(successful), total),
            'Fast1': _ratio(sum(1 for row in rows if row.get('paper_fast1')), total),
            'Speed': (sum(speed_metric_values) / total) if total > 0 else None,
        },
        'paper_metrics_by_mode': {
            mode: {
                'Success': _ratio(len(successful), total),
                'Fast1': _ratio(
                    sum(
                        1
                        for row in rows
                        if row.get('best_correct') and isinstance(row.get(f'{mode}_speedup'), (int, float)) and float(row[f'{mode}_speedup']) >= 1.0
                    ),
                    total,
                ),
                'Speed': (
                    sum(
                        float(row[f'{mode}_speedup'])
                        if row.get('best_correct') and isinstance(row.get(f'{mode}_speedup'), (int, float))
                        else 0.0
                        for row in rows
                    )
                    / total
                    if total > 0
                    else None
                ),
            }
            for mode in ('torch_eager', 'torch_compile_default', 'torch_compile_max_autotune')
        },
    }


def write_batch_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        'alias', 'level', 'problem_id', 'backend', 'workflow', 'run_profile', 'search_profile', 'evaluator_profile',
        'measurement_profile', 'preset', 'evaluation_profile', 'kernelbench_evaluator', 'status', 'task_name',
        'best_node_id', 'best_status', 'best_node_is_root', 'best_correct', 'paper_fast1', 'root_correct',
        'non_root_correct', 'improved_over_reference', 'candidate_runtime', 'reference_runtime', 'speedup',
        'primary_reference', 'torch_eager_reference_runtime', 'torch_compile_default_reference_runtime',
        'torch_compile_max_autotune_reference_runtime', 'torch_eager_speedup', 'torch_compile_default_speedup',
        'torch_compile_max_autotune_speedup', 'candidate_total_count', 'candidate_compile_count',
        'candidate_correct_count', 'compile_rate', 'correct_rate', 'failure_stage', 'failure_type',
        'validation_correctness_matches', 'validation_speed_direction_matches', 'run_path', 'validation_path', 'error',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def runtime_for_mode(mapping: dict[str, float | None] | None, mode: str, fallback: float | None = None) -> float | None:
    if mapping and mode in mapping:
        return mapping.get(mode)
    return fallback


def speedup_for_mode(mapping: dict[str, float | None] | None, mode: str, fallback: float | None = None) -> float | None:
    if mapping and mode in mapping:
        return mapping.get(mode)
    return fallback


def candidate_attempt_stats(result) -> dict[str, float | int | None]:
    candidates = [node for node_id, node in result.nodes.items() if node_id != 'root']
    total = len(candidates)
    compile_count = sum(1 for node in candidates if node.compile_ok)
    correct_count = sum(1 for node in candidates if node.correct)
    return {
        'total': total,
        'compile': compile_count,
        'correct': correct_count,
        'compile_rate': _ratio(compile_count, total),
        'correct_rate': _ratio(correct_count, total),
    }


def format_speedup(value: float | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.3f}x'


def _normalize_task(item: dict[str, Any], default_backend: str | None = None) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError('Each manifest task must be an object.')
    if 'level' not in item or 'problem_id' not in item:
        raise ValueError('Each manifest task must define level and problem_id.')
    level = int(item['level'])
    problem_id = int(item['problem_id'])
    return {
        'alias': item.get('alias') or f'L{level}_P{problem_id}',
        'level': level,
        'problem_id': problem_id,
        'backend': item.get('backend', default_backend or 'triton'),
    }


def _discover_all_official_tasks(kernelbench_root: Path, backend: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        level_dir = kernelbench_root / 'KernelBench' / f'level{level}'
        if not level_dir.exists():
            continue
        for path in sorted(level_dir.glob('*.py')):
            name = path.stem
            prefix, _, title = name.partition('_')
            if not prefix.isdigit():
                continue
            problem_id = int(prefix)
            alias = f'L{level}_P{problem_id}_{title}' if title else f'L{level}_P{problem_id}'
            tasks.append({
                'alias': alias,
                'level': level,
                'problem_id': problem_id,
                'backend': backend,
            })
    return tasks


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
