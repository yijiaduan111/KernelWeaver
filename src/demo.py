from __future__ import annotations

from textwrap import dedent

from .models import StrategySpec, TaskSpec, TestCase


def build_demo_tasks() -> list[TaskSpec]:
    square_task = TaskSpec(
        name="square_list",
        description="Single-operator task that squares each element in a list.",
        source_code=dedent(
            """
            def solve(data):
                # <<<IMPROVE:body>>>
                result = []
                for value in data:
                    result.append(value * value)
                return result
                # <<<END_IMPROVE>>>
            """
        ).strip()
        + "\n",
        reference_code=dedent(
            """
            def reference_solve(data):
                return [value * value for value in data]
            """
        ).strip()
        + "\n",
        function_name="solve",
        reference_function_name="reference_solve",
        test_cases=[
            TestCase(label="small", args=[[1, 2, 3, 4]]),
            TestCase(label="mixed", args=[[-2, 0, 5, 9]]),
        ],
        benchmark_cases=[
            TestCase(label="bench-a", args=[list(range(512))]),
            TestCase(label="bench-b", args=[list(range(-256, 256))]),
        ],
        tags=["single-op", "python"],
        strategy_catalog=[
            StrategySpec(
                name="list_comprehension",
                anchor_name="body",
                strategy_summary="Replace the append loop with a single list comprehension.",
                instruction="Rewrite the loop body as a direct list comprehension to reduce Python overhead.",
                expected_gain="Fewer Python bytecode operations per element.",
                good_body="return [value * value for value in data]\n",
                broken_body="return [value * value for value in data\n",
                debug_body="return [value * value for value in data]\n",
                broken_failure_type="compile_error",
            ),
            StrategySpec(
                name="map_lambda",
                anchor_name="body",
                strategy_summary="Use map with a lambda to avoid manual append bookkeeping.",
                instruction="Replace the explicit loop with a mapped transformation.",
                expected_gain="Slightly smaller Python loop body.",
                good_body="return list(map(lambda value: value * value, data))\n",
            ),
        ],
    )

    fused_task = TaskSpec(
        name="fused_affine_relu",
        description="Fused multi-step task that combines bias, scale, and ReLU in a single pass.",
        source_code=dedent(
            """
            def solve(data, bias, scale):
                # <<<IMPROVE:body>>>
                shifted = []
                for value in data:
                    shifted.append(value + bias)
                scaled = []
                for value in shifted:
                    scaled.append(value * scale)
                result = []
                for value in scaled:
                    result.append(value if value > 0 else 0.0)
                return result
                # <<<END_IMPROVE>>>
            """
        ).strip()
        + "\n",
        reference_code=dedent(
            """
            def reference_solve(data, bias, scale):
                return [max((value + bias) * scale, 0.0) for value in data]
            """
        ).strip()
        + "\n",
        function_name="solve",
        reference_function_name="reference_solve",
        test_cases=[
            TestCase(label="simple", args=[[1.0, -2.0, 3.0], 0.5, 2.0]),
            TestCase(label="mixed", args=[[-3.0, 2.5, 0.0, 6.0], -1.0, 1.5]),
        ],
        benchmark_cases=[
            TestCase(label="bench-a", args=[[float(index) for index in range(-256, 256)], 0.75, 1.25]),
            TestCase(label="bench-b", args=[[float(index) / 5 for index in range(-512, 512)], -0.5, 0.9]),
        ],
        tags=["fused-op", "python"],
        strategy_catalog=[
            StrategySpec(
                name="single_pass_fusion",
                anchor_name="body",
                strategy_summary="Fuse the three passes into one loop.",
                instruction="Compute bias, scale, and ReLU together in a single loop.",
                expected_gain="Eliminate intermediate lists and three-pass overhead.",
                good_body=dedent(
                    """
                    result = []
                    for value in data:
                        fused = (value + bias) * scale
                        result.append(fused if fused > 0 else 0.0)
                    return result
                    """
                ),
                broken_body=dedent(
                    """
                    result = []
                    for value in data:
                        fused = value + bias * scale
                        result.append(fused if fused > 0 else 0.0)
                    return result
                    """
                ),
                debug_body=dedent(
                    """
                    result = []
                    for value in data:
                        fused = (value + bias) * scale
                        result.append(fused if fused > 0 else 0.0)
                    return result
                    """
                ),
                broken_failure_type="correctness_error",
            ),
            StrategySpec(
                name="fused_comprehension",
                anchor_name="body",
                strategy_summary="Use a single comprehension for the whole fused computation.",
                instruction="Rewrite the kernel as one fused list comprehension.",
                expected_gain="Collapse the computation into one expression.",
                good_body="return [max((value + bias) * scale, 0.0) for value in data]\n",
            ),
        ],
    )
    return [square_task, fused_task]
