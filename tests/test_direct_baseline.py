import tempfile
import unittest
from pathlib import Path

from stark.direct_baseline import extract_python_code, run_direct_baseline
from stark.models import EvaluationResult, StarkConfig, TaskSpec


class DirectBaselineTests(unittest.TestCase):
    def test_extract_prefers_modelnew_fenced_block(self):
        response = """
```python
class Other:
    pass
```
```python
class ModelNew:
    pass
```
"""
        self.assertEqual(extract_python_code(response), "class ModelNew:\n    pass\n")

    def test_runner_records_one_candidate_without_framework_profiles(self):
        task = TaskSpec(
            name="mock_task",
            description="mock",
            source_code="class ModelNew:\n    pass\n",
            reference_code="class Model:\n    pass\n",
            function_name="ModelNew",
            reference_function_name="Model",
            test_cases=[],
            benchmark_cases=[],
            benchmark_family="kernelbench",
            entry_kind="model_class",
            level=1,
            problem_id=1,
            backend="cuda",
        )
        config = StarkConfig(run_profile="main", search_profile="main", evaluator_profile="main", measurement_profile="main")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_direct_baseline(
                task,
                config,
                _FakeProvider(),
                _FakeEvaluator(),
                artifact_dir=tmpdir,
                provider_name="fake",
                model_name="fake-model",
            )
            self.assertEqual(result.workflow, "direct_llm_baseline")
            self.assertEqual(result.best_node_id, "candidate_1")
            self.assertEqual(result.stats["attempt_count"], 1)
            self.assertIsNone(result.semantic_profile)
            self.assertIsNone(result.strategy_portfolio)
            self.assertEqual(result.grounded_regions, [])
            self.assertTrue((Path(tmpdir) / "raw_response.txt").exists())
            self.assertTrue((Path(tmpdir) / "candidate.py").exists())
            self.assertTrue((Path(tmpdir) / "direct_baseline.json").exists())


class _FakeProvider:
    name = "fake"

    def generate_text(self, system_prompt, user_payload, temperature=0.2, purpose="generic"):
        self.last_payload = user_payload
        return "```python\nclass ModelNew:\n    pass\n```"


class _FakeEvaluator:
    def evaluate(self, task, code, config):
        return EvaluationResult(
            compile_ok="class ModelNew" in code,
            correct="class ModelNew" in code,
            runtime=1.0,
            score=1.0,
            reference_runtime=2.0,
            speedup=2.0,
            reference_runtimes={"torch_eager": 2.0},
            speedups={"torch_eager": 2.0},
            primary_reference="torch_eager",
        )


if __name__ == "__main__":
    unittest.main()
