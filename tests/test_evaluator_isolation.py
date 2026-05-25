from src.evaluation.base import Evaluator
from src.evaluation.isolated import IsolatedEvaluator, _evaluation_from_payload
from src.models import EvaluationResult, StarkConfig, TaskSpec


class FakeEvaluator(Evaluator):
    def evaluate(self, task, code, config):
        return EvaluationResult(
            compile_ok=True,
            correct=True,
            runtime=1.0,
            score=1.0,
            logs=["fake"],
            reference_runtime=2.0,
            speedup=2.0,
            reference_runtimes={"torch_eager": 2.0},
            speedups={"torch_eager": 2.0},
            primary_reference="torch_eager",
        )


def _task():
    return TaskSpec(
        name="t",
        description="",
        source_code="",
        reference_code="",
        function_name="ModelNew",
        reference_function_name="Model",
        test_cases=[],
        benchmark_cases=[],
    )


def test_isolated_evaluator_passthrough_when_disabled():
    result = IsolatedEvaluator(FakeEvaluator()).evaluate(_task(), "code", StarkConfig(evaluator_isolation="off"))
    assert result.correct
    assert result.speedup == 2.0
    assert result.logs == ["fake"]


def test_evaluation_payload_roundtrip_preserves_metrics():
    payload = {
        "compile_ok": True,
        "correct": True,
        "runtime": 1.5,
        "score": 1.5,
        "logs": ["ok"],
        "failure_type": None,
        "failure_stage": "none",
        "reference_runtime": 3.0,
        "speedup": 2.0,
        "reference_runtimes": {"torch_eager": 3.0},
        "speedups": {"torch_eager": 2.0},
        "primary_reference": "torch_eager",
    }
    result = _evaluation_from_payload(payload)
    assert result.compile_ok
    assert result.correct
    assert result.speedup == 2.0
    assert result.reference_runtimes["torch_eager"] == 3.0
