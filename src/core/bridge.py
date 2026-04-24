"""Bridge external tasks into STARK's internal task model.

The most important path in this file is the KernelBench bridge. It reads
official benchmark problems from a read-only external clone and converts
them into anchored `TaskSpec` objects that can be consumed by the same
workflow used for demo and Triton tasks.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import GroundedRegion, StrategySpec, TaskSpec, TestCase


class BridgeLoadError(ValueError):
    pass


@dataclass(slots=True)
class BridgeTaskConfig:
    name: str
    description: str
    source_path: str | Path
    reference_path: str | Path
    function_name: str
    reference_function_name: str
    test_cases: list[TestCase]
    benchmark_cases: list[TestCase]
    tags: list[str] = field(default_factory=list)
    strategy_catalog: list[StrategySpec] = field(default_factory=list)
    source_origin: str | None = None
    benchmark_family: str = "kernelbench"


@dataclass(slots=True)
class OfficialProblemInfo:
    path: Path
    source_code: str
    description: str
    imports_block: str
    class_preamble: str
    init_signature: str
    init_body: str
    forward_signature: str
    forward_body: str
    forward_steps: list[str] = field(default_factory=list)


from .bridge_specs import (
    CUDA_ENABLED_TARGETS as _CUDA_ENABLED_TARGETS,
    NATIVE_CUDA_TARGETS as _NATIVE_CUDA_TARGETS,
    SAFE_CUDA_FORWARD_ONLY_TARGETS as _SAFE_CUDA_FORWARD_ONLY_TARGETS,
    SELECTED_TARGETS as _SELECTED_TARGETS,
    selected_kernelbench_targets,
)


def _region_role(anchor_name: str) -> str:
    if anchor_name.startswith("forward_step_"):
        return "forward_step"
    return anchor_name

class KernelBenchTaskBridge:
    def load_task(self, config: BridgeTaskConfig) -> TaskSpec:
        """Load a local hand-written callable task into a `TaskSpec`."""
        if not config.test_cases:
            raise BridgeLoadError(f"Bridge task '{config.name}' must define at least one test case.")
        if not config.benchmark_cases:
            raise BridgeLoadError(f"Bridge task '{config.name}' must define at least one benchmark case.")
        source_path = Path(config.source_path)
        reference_path = Path(config.reference_path)
        source_code = self._load_python_source(source_path, config.function_name, label="candidate")
        reference_code = self._load_python_source(reference_path, config.reference_function_name, label="reference")
        return TaskSpec(
            name=config.name,
            description=config.description,
            source_code=source_code,
            reference_code=reference_code,
            function_name=config.function_name,
            reference_function_name=config.reference_function_name,
            test_cases=list(config.test_cases),
            benchmark_cases=list(config.benchmark_cases),
            tags=list(config.tags),
            strategy_catalog=list(config.strategy_catalog),
            source_origin=config.source_origin or str(source_path),
            benchmark_family=config.benchmark_family,
            entry_kind="callable",
        )

    def load_official_problem(
        self,
        kernelbench_root: str | Path,
        level: int,
        problem_id: int,
        backend: str = "triton",
    ) -> TaskSpec:
        """Load one official KernelBench problem and build an anchored scaffold.

        The external benchmark file remains untouched. STARK extracts the
        official `Model` structure, synthesizes a `ModelNew` scaffold, and
        injects grounded edit anchors so the agents can make local changes
        without rewriting the full module arbitrarily.
        """
        if backend not in {"triton", "cuda"}:
            raise BridgeLoadError(f"Unsupported KernelBench backend: {backend}. Supported backends: triton, cuda.")
        root = Path(kernelbench_root)
        problem_path = self._resolve_problem_path(root, level, problem_id)
        info = self._inspect_official_problem(problem_path)
        target = _SELECTED_TARGETS.get((level, problem_id))
        is_auto_bridge = target is None
        if is_auto_bridge:
            target = self._build_auto_target(problem_path, info, level, problem_id, backend)
            test_cases, benchmark_cases = self._build_auto_cases(problem_path)
        else:
            test_cases = self._build_cases(target, kind="test")
            benchmark_cases = self._build_cases(target, kind="benchmark")
        scaffold = self._build_model_scaffold(
            info,
            level=level,
            backend=backend,
            level_problem=(level, problem_id),
            target=target,
            is_auto_bridge=is_auto_bridge,
        )
        grounded_regions = self._extract_grounded_regions(scaffold)
        tags = [backend if tag == "triton" else tag for tag in target["tags"]]
        if backend == "cuda":
            if self._uses_native_cuda_scaffold((level, problem_id)):
                if "native_cuda" not in tags:
                    tags.append("native_cuda")
            else:
                if "cuda_safe_forward_only" not in tags:
                    tags.append("cuda_safe_forward_only")
        strategy_catalog = self._strategy_catalog_for_backend((level, problem_id), backend, target.get("strategies", []))
        return TaskSpec(
            name=target["task_name"],
            description=f"{target['title']} (official KernelBench task with reduced local evaluation profile)",
            source_code=scaffold,
            reference_code=info.source_code,
            function_name="ModelNew",
            reference_function_name="Model",
            test_cases=test_cases,
            benchmark_cases=benchmark_cases,
            tags=tags,
            strategy_catalog=strategy_catalog,
            source_origin=str(problem_path),
            benchmark_family="kernelbench",
            entry_kind="model_class",
            level=level,
            problem_id=problem_id,
            backend=backend,
            source_root=str(root),
            grounded_regions=grounded_regions,
        )

    def _resolve_problem_path(self, kernelbench_root: Path, level: int, problem_id: int) -> Path:
        candidate_dirs = [
            kernelbench_root / "KernelBench" / f"level{level}",
            kernelbench_root / f"level{level}",
        ]
        existing_dirs = [path for path in candidate_dirs if path.exists()]
        if not existing_dirs:
            locations = ", ".join(str(path) for path in candidate_dirs)
            raise BridgeLoadError(f"KernelBench level directory does not exist. Checked: {locations}")
        for level_dir in existing_dirs:
            matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
            if matches:
                return matches[0]
        searched = ", ".join(str(path) for path in existing_dirs)
        raise BridgeLoadError(f"KernelBench problem L{level}/P{problem_id} was not found under: {searched}")

    def _build_auto_target(
        self,
        problem_path: Path,
        info: OfficialProblemInfo,
        level: int,
        problem_id: int,
        backend: str,
    ) -> dict[str, Any]:
        stem = problem_path.stem
        slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
        title = info.description.strip() or stem.replace("_", " ")
        tags = self._infer_auto_tags(level, stem, title, backend)
        return {
            "alias": f"AUTO-L{level}-P{problem_id}",
            "task_name": f"kernelbench_l{level}_{problem_id}_{slug}",
            "title": f"KernelBench Level {level} / {problem_id} {title}",
            "tags": tags,
            "strategies": [],
        }

    def _build_auto_cases(self, problem_path: Path) -> tuple[list[TestCase], list[TestCase]]:
        module = self._load_runtime_module(problem_path)
        init_args = self._normalize_runtime_values(self._call_runtime_factory(module, "get_init_inputs", problem_path))
        benchmark_args = self._normalize_runtime_values(self._call_runtime_factory(module, "get_inputs", problem_path))
        benchmark_case = self._make_auto_case(
            module,
            init_args,
            benchmark_args,
            label="benchmark-1",
            budget=262_144,
        )
        test_case = self._make_auto_case(
            module,
            init_args,
            benchmark_args,
            label="test-1",
            budget=65_536,
            fallback_case=benchmark_case,
        )
        return [test_case], [benchmark_case]

    def _make_auto_case(
        self,
        module: Any,
        init_args: list[Any],
        args: list[Any],
        label: str,
        budget: int,
        fallback_case: TestCase | None = None,
    ) -> TestCase:
        reduced_init_args = self._clone_case_values(init_args)
        reduced_args = self._reduce_case_arguments(args, reduced_init_args, budget=budget)
        if self._validate_runtime_case(module, reduced_init_args, reduced_args):
            return TestCase(
                label=label,
                args=reduced_args,
                kwargs={},
                init_args=reduced_init_args,
                init_kwargs={},
            )
        if fallback_case is not None:
            return TestCase(
                label=label,
                args=self._clone_case_values(fallback_case.args),
                kwargs=dict(fallback_case.kwargs),
                init_args=self._clone_case_values(fallback_case.init_args),
                init_kwargs=dict(fallback_case.init_kwargs),
            )
        return TestCase(
            label=label,
            args=self._clone_case_values(args),
            kwargs={},
            init_args=self._clone_case_values(init_args),
            init_kwargs={},
        )

    def _inspect_official_problem(self, path: Path) -> OfficialProblemInfo:
        """Extract the parts of an official benchmark module needed for scaffolding."""
        source = path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno}:{exc.offset}"
            raise BridgeLoadError(f"Official KernelBench source has invalid syntax at {location}: {exc.msg}") from exc
        model_class = self._find_class(module, "Model", path)
        self._find_function(module, "get_inputs", path)
        self._find_function(module, "get_init_inputs", path)
        init_node = self._find_method(model_class, "__init__", path)
        forward_node = self._find_method(model_class, "forward", path)
        imports_block = self._extract_imports(source, module)
        class_preamble = self._extract_class_preamble(source, model_class)
        init_signature = self._extract_signature(source, init_node)
        forward_signature = self._extract_signature(source, forward_node)
        init_body = self._extract_body(source, init_node, drop_super_init=True)
        forward_body = self._extract_body(source, forward_node, drop_super_init=False)
        forward_steps = self._extract_forward_steps(source, forward_node)
        description = ast.get_docstring(model_class) or path.stem.replace("_", " ")
        return OfficialProblemInfo(
            path=path,
            source_code=source,
            description=description,
            imports_block=imports_block,
            class_preamble=class_preamble,
            init_signature=init_signature,
            init_body=init_body,
            forward_signature=forward_signature,
            forward_body=forward_body,
            forward_steps=forward_steps,
        )

    @staticmethod
    def _find_class(module: ast.Module, class_name: str, path: Path) -> ast.ClassDef:
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        raise BridgeLoadError(f"Class '{class_name}' was not found in {path}")

    @staticmethod
    def _find_function(module: ast.Module, function_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return node
        raise BridgeLoadError(f"Function '{function_name}' was not found in {path}")

    @staticmethod
    def _find_method(class_node: ast.ClassDef, method_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        raise BridgeLoadError(f"Method '{method_name}' was not found in class '{class_node.name}' from {path}")

    @staticmethod
    def _extract_imports(source: str, module: ast.Module) -> str:
        chunks: list[str] = []
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    chunks.append(segment.strip())
        return "\n".join(chunks)

    @staticmethod
    def _extract_signature(source: str, function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        segment = ast.get_source_segment(source, function_node)
        if not segment:
            raise BridgeLoadError(f"Failed to extract signature for function '{function_node.name}'")
        first_line = segment.splitlines()[0].strip()
        if not first_line.startswith("def ") or not first_line.endswith(":"):
            raise BridgeLoadError(f"Unsupported function header for '{function_node.name}': {first_line}")
        return first_line[4:-1].strip()

    @staticmethod
    def _extract_body(
        source: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        drop_super_init: bool,
    ) -> str:
        statements = list(function_node.body)
        if statements and isinstance(statements[0], ast.Expr) and isinstance(getattr(statements[0], "value", None), ast.Constant):
            if isinstance(statements[0].value.value, str):
                statements = statements[1:]
        chunks: list[str] = []
        for statement in statements:
            if drop_super_init and _is_super_init_statement(statement):
                continue
            segment = ast.get_source_segment(source, statement)
            if segment:
                chunks.append(segment)
        return textwrap.dedent("\n".join(chunks)).strip("\n")

    @staticmethod
    def _extract_class_preamble(source: str, class_node: ast.ClassDef) -> str:
        chunks: list[str] = []
        for statement in class_node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(statement, ast.Expr) and isinstance(getattr(statement, "value", None), ast.Constant):
                if isinstance(statement.value.value, str):
                    continue
            segment = ast.get_source_segment(source, statement)
            if segment:
                chunks.append(textwrap.dedent(segment).strip("\n"))
        return "\n".join(chunks).strip("\n")

    @staticmethod
    def _extract_forward_steps(
        source: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Split a forward body into statement-sized editable steps.

        Level 3 tasks benefit from finer-grained anchors because their
        forward methods are usually multi-stage blocks instead of one
        compact expression.
        """
        statements = list(function_node.body)
        if statements and isinstance(statements[0], ast.Expr) and isinstance(getattr(statements[0], "value", None), ast.Constant):
            if isinstance(statements[0].value.value, str):
                statements = statements[1:]
        steps: list[str] = []
        for statement in statements:
            segment = ast.get_source_segment(source, statement)
            if segment:
                steps.append(textwrap.dedent(segment).strip("\n"))
        if len(steps) == 1 and statements and isinstance(statements[0], ast.Return):
            return KernelBenchTaskBridge._split_single_return_step(source, statements[0])
        return steps

    @staticmethod
    def _split_single_return_step(source: str, statement: ast.Return) -> list[str]:
        """Turn a one-line return into two editable steps for Level 3 scaffolds."""
        if statement.value is None:
            return ["return None"]
        expression = ast.get_source_segment(source, statement.value)
        if not expression:
            return ["return None"]
        temp_name = "_stark_forward_value"
        return [f"{temp_name} = {expression}", f"return {temp_name}"]


    def _build_model_scaffold(
        self,
        info: OfficialProblemInfo,
        level: int,
        backend: str,
        level_problem: tuple[int, int],
        target: dict[str, Any] | None = None,
        is_auto_bridge: bool = False,
    ) -> str:
        """Render the anchored `ModelNew` module used by the STARK agents."""
        if backend == "cuda":
            return self._build_cuda_model_scaffold(
                info,
                level_problem,
                target=target,
                is_auto_bridge=is_auto_bridge,
            )

        imports = info.imports_block.strip()
        parts = []
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(
            [
                "# <<<IMPROVE:helpers>>>",
                "# <<<END_IMPROVE>>>",
                "",
                "class ModelNew(nn.Module):",
            ]
        )
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend(
            [
                f"    def {info.init_signature}:",
                "        super().__init__()",
                "        # <<<IMPROVE:init_body>>>",
            ]
        )
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
                f"    def {info.forward_signature}:",
            ]
        )
        forward_steps = self._curated_forward_steps(info, target, allow_level3_steps=level >= 3)
        if forward_steps:
            self._append_forward_step_regions(parts, forward_steps, indent="        ")
        else:
            parts.append("        # <<<IMPROVE:forward_body>>>")
            if info.forward_body:
                for line in info.forward_body.splitlines():
                    parts.append(f"        {line}" if line else "")
            parts.append("        # <<<END_IMPROVE>>>")
        parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def _build_cuda_model_scaffold(
        self,
        info: OfficialProblemInfo,
        level_problem: tuple[int, int],
        target: dict[str, Any] | None = None,
        is_auto_bridge: bool = False,
    ) -> str:
        if self._uses_safe_cuda_scaffold(level_problem, is_auto_bridge=is_auto_bridge):
            return self._build_safe_cuda_model_scaffold(info, target=target)

        imports = self._ensure_cuda_extension_imports(info.imports_block.strip())
        helper_body, cpp_body, cu_body, forward_body = self._cuda_backend_bodies(level_problem, info.forward_body)
        parts = []
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(
            [
                "# <<<IMPROVE:helpers>>>",
            ]
        )
        for line in helper_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                "",
                'CUDA_CPP_SRC = r"""',
                "# <<<IMPROVE:cuda_cpp>>>",
            ]
        )
        for line in cpp_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                '"""',
                "",
                'CUDA_CU_SRC = r"""',
                "# <<<IMPROVE:cuda_cu>>>",
            ]
        )
        for line in cu_body.splitlines():
            parts.append(line)
        parts.extend(
            [
                "# <<<END_IMPROVE>>>",
                '"""',
                "",
                "class ModelNew(nn.Module):",
            ]
        )
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend(
            [
                f"    def {info.init_signature}:",
                "        super().__init__()",
                "        # <<<IMPROVE:init_body>>>",
            ]
        )
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
                f"    def {info.forward_signature}:",
                "        # <<<IMPROVE:forward_body>>>",
            ]
        )
        for line in forward_body.splitlines():
            parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
            ]
        )
        return "\n".join(parts).rstrip() + "\n"

    @staticmethod
    def _curated_forward_steps(
        info: OfficialProblemInfo,
        target: dict[str, Any] | None,
        allow_level3_steps: bool,
    ) -> list[str]:
        if target is not None:
            curated_steps = target.get("forward_steps")
            if curated_steps:
                return [str(step).strip("\n") for step in curated_steps]
        if allow_level3_steps and info.forward_steps:
            return list(info.forward_steps)
        return []

    @staticmethod
    def _append_forward_step_regions(parts: list[str], steps: list[str], indent: str) -> None:
        for index, step in enumerate(steps, start=1):
            parts.append(f"{indent}# <<<IMPROVE:forward_step_{index}>>>")
            if step:
                for line in step.splitlines():
                    parts.append(f"{indent}{line}" if line else "")
            parts.append(f"{indent}# <<<END_IMPROVE>>>")

    @staticmethod
    def _ensure_cuda_extension_imports(imports: str) -> str:
        lines = [line for line in imports.splitlines() if line.strip()]
        required = [
            "import hashlib",
            "from torch.utils.cpp_extension import load_inline",
        ]
        existing = set(line.strip() for line in lines)
        for line in required:
            if line not in existing:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _uses_native_cuda_scaffold(level_problem: tuple[int, int]) -> bool:
        return level_problem in _NATIVE_CUDA_TARGETS

    @staticmethod
    def _uses_safe_cuda_scaffold(level_problem: tuple[int, int], is_auto_bridge: bool = False) -> bool:
        return is_auto_bridge or level_problem in _SAFE_CUDA_FORWARD_ONLY_TARGETS

    def _build_safe_cuda_model_scaffold(self, info: OfficialProblemInfo, target: dict[str, Any] | None = None) -> str:
        imports = info.imports_block.strip()
        parts = []
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(
            [
                "# <<<IMPROVE:helpers>>>",
                "# Keep this CUDA task on a safe Python scaffold until a task-specific native kernel bridge is added.",
                "# Do not add load_inline / pybind / handwritten CUDA here unless the task already has a curated native scaffold.",
                "# <<<END_IMPROVE>>>",
                "",
                "class ModelNew(nn.Module):",
            ]
        )
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend(
            [
                f"    def {info.init_signature}:",
                "        super().__init__()",
                "        # <<<IMPROVE:init_body>>>",
            ]
        )
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(
            [
                "        # <<<END_IMPROVE>>>",
                "",
                f"    def {info.forward_signature}:",
            ]
        )
        forward_steps = self._curated_forward_steps(info, target, allow_level3_steps=False)
        if forward_steps:
            self._append_forward_step_regions(parts, forward_steps, indent="        ")
        else:
            parts.append("        # <<<IMPROVE:forward_body>>>")
            if info.forward_body:
                for line in info.forward_body.splitlines():
                    parts.append(f"        {line}" if line else "")
            else:
                parts.append("        raise NotImplementedError('Empty forward body for safe CUDA scaffold')")
            parts.append("        # <<<END_IMPROVE>>>")
        parts.append("")
        return "\n".join(parts).rstrip() + "\n"
    def _cuda_backend_bodies(self, level_problem: tuple[int, int], baseline_forward_body: str) -> tuple[str, str, str, str]:
        if level_problem == (1, 25):
            return (
                textwrap.dedent(
                    """
                    _STARK_EXTENSION = None

                    def _stark_strip_anchor_markers(source: str) -> str:
                        cleaned_lines = []
                        for line in source.splitlines():
                            stripped = line.lstrip()
                            if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                                continue
                            cleaned_lines.append(line)
                        return "\\n".join(cleaned_lines)

                    def _stark_extension_name() -> str:
                        digest = hashlib.sha1(
                            (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                        ).hexdigest()[:12]
                        return f"stark_cuda_swish_{digest}"

                    def _stark_get_extension():
                        global _STARK_EXTENSION
                        if _STARK_EXTENSION is None:
                            _STARK_EXTENSION = load_inline(
                                name=_stark_extension_name(),
                                cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                                cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                                functions=None,
                                extra_cflags=["-O3"],
                                extra_cuda_cflags=["-O3", "--use_fast_math"],
                                with_cuda=True,
                                verbose=False,
                            )
                        return _STARK_EXTENSION
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>

                    torch::Tensor swish_cuda(torch::Tensor x);

                    torch::Tensor swish_forward(torch::Tensor x) {
                        return swish_cuda(x);
                    }

                    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                        m.def("swish_cuda", &swish_forward, "Swish forward (CUDA)");
                    }
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>
                    #include <cuda.h>
                    #include <cuda_runtime.h>

                    template <typename scalar_t>
                    __global__ void swish_kernel(const scalar_t* x, scalar_t* out, int64_t n) {
                        int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
                        if (index < n) {
                            scalar_t value = x[index];
                            scalar_t sigmoid = scalar_t(1) / (scalar_t(1) + exp(-value));
                            out[index] = value * sigmoid;
                        }
                    }

                    torch::Tensor swish_cuda(torch::Tensor x) {
                        TORCH_CHECK(x.is_cuda(), "swish_cuda: expected a CUDA tensor");
                        auto input = x.contiguous();
                        auto output = torch::empty_like(input);
                        int64_t n = input.numel();
                        constexpr int threads = 256;
                        const int blocks = static_cast<int>((n + threads - 1) / threads);
                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "swish_cuda", [&] {
                            swish_kernel<scalar_t><<<blocks, threads>>>(
                                input.data_ptr<scalar_t>(),
                                output.data_ptr<scalar_t>(),
                                n
                            );
                        });
                        return output.view(input.sizes());
                    }
                    """
                ).strip("\n"),
                "return _stark_get_extension().swish_cuda(x)\n",
            )
        if level_problem == (1, 47):
            return (
                textwrap.dedent(
                    """
                    _STARK_EXTENSION = None

                    def _stark_strip_anchor_markers(source: str) -> str:
                        cleaned_lines = []
                        for line in source.splitlines():
                            stripped = line.lstrip()
                            if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                                continue
                            cleaned_lines.append(line)
                        return "\\n".join(cleaned_lines)

                    def _stark_extension_name() -> str:
                        digest = hashlib.sha1(
                            (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                        ).hexdigest()[:12]
                        return f"stark_cuda_sumdim1_{digest}"

                    def _stark_get_extension():
                        global _STARK_EXTENSION
                        if _STARK_EXTENSION is None:
                            _STARK_EXTENSION = load_inline(
                                name=_stark_extension_name(),
                                cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                                cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                                functions=None,
                                extra_cflags=["-O3"],
                                extra_cuda_cflags=["-O3", "--use_fast_math"],
                                with_cuda=True,
                                verbose=False,
                            )
                        return _STARK_EXTENSION
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>

                    torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x);

                    torch::Tensor sum_dim1_keepdim_forward(torch::Tensor x) {
                        return sum_dim1_keepdim_cuda(x);
                    }

                    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                        m.def("sum_dim1_keepdim_cuda", &sum_dim1_keepdim_forward, "Sum keepdim over dim=1 (CUDA)");
                    }
                    """
                ).strip("\n"),
                textwrap.dedent(
                    """
                    #include <torch/extension.h>
                    #include <cuda.h>
                    #include <cuda_runtime.h>

                    template <typename scalar_t>
                    __global__ void sum_dim1_keepdim_kernel(
                        const scalar_t* x,
                        scalar_t* out,
                        int64_t batch,
                        int64_t channels,
                        int64_t width
                    ) {
                        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
                        int64_t total = batch * width;
                        if (linear < total) {
                            int64_t batch_index = linear / width;
                            int64_t width_index = linear % width;
                            scalar_t acc = scalar_t(0);
                            int64_t base = batch_index * channels * width + width_index;
                            for (int64_t channel = 0; channel < channels; ++channel) {
                                acc += x[base + channel * width];
                            }
                            out[batch_index * width + width_index] = acc;
                        }
                    }

                    torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x) {
                        TORCH_CHECK(x.is_cuda(), "sum_dim1_keepdim_cuda: expected a CUDA tensor");
                        TORCH_CHECK(x.dim() == 3, "sum_dim1_keepdim_cuda: expected a 3D tensor");
                        auto input = x.contiguous();
                        auto output = torch::zeros({input.size(0), 1, input.size(2)}, input.options());
                        const int64_t batch = input.size(0);
                        const int64_t channels = input.size(1);
                        const int64_t width = input.size(2);
                        const int64_t total = batch * width;
                        constexpr int threads = 256;
                        const int blocks = static_cast<int>((total + threads - 1) / threads);
                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_dim1_keepdim_cuda", [&] {
                            sum_dim1_keepdim_kernel<scalar_t><<<blocks, threads>>>(
                                input.data_ptr<scalar_t>(),
                                output.data_ptr<scalar_t>(),
                                batch,
                                channels,
                                width
                            );
                        });
                        return output;
                    }
                    """
                ).strip("\n"),
                "return _stark_get_extension().sum_dim1_keepdim_cuda(x)\n",
            )
        return self._generic_cuda_backend_bodies(level_problem, baseline_forward_body)

    @staticmethod
    def _generic_cuda_backend_bodies(level_problem: tuple[int, int], baseline_forward_body: str) -> tuple[str, str, str, str]:
        level, problem_id = level_problem
        helper_body = textwrap.dedent(
            f"""
            _STARK_EXTENSION = None

            def _stark_strip_anchor_markers(source: str) -> str:
                cleaned_lines = []
                for line in source.splitlines():
                    stripped = line.lstrip()
                    if stripped.startswith("# <<<IMPROVE:") or stripped.startswith("# <<<END_IMPROVE>>>"):
                        continue
                    cleaned_lines.append(line)
                return "\\n".join(cleaned_lines)

            def _stark_extension_name() -> str:
                digest = hashlib.sha1(
                    (_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode("utf-8")
                ).hexdigest()[:12]
                return f"stark_cuda_l{level}_p{problem_id}_{{digest}}"

            def _stark_get_extension():
                global _STARK_EXTENSION
                if _STARK_EXTENSION is None:
                    _STARK_EXTENSION = load_inline(
                        name=_stark_extension_name(),
                        cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),
                        cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),
                        functions=None,
                        extra_cflags=["-O3"],
                        extra_cuda_cflags=["-O3", "--use_fast_math"],
                        with_cuda=True,
                        verbose=False,
                    )
                return _STARK_EXTENSION
            """
        ).strip("\n")
        cpp_body = textwrap.dedent(
            """
            #include <torch/extension.h>

            // Replace this anchor with your pybind exports for custom CUDA entrypoints.
            // Example:
            // torch::Tensor custom_cuda(torch::Tensor x);
            // PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            //     m.def("custom_cuda", &custom_cuda, "Custom CUDA op");
            // }

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
            """
        ).strip("\n")
        cu_body = textwrap.dedent(
            """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            // Replace this anchor with custom CUDA kernels plus their exported wrapper functions.
            // The Python forward body can call them via _stark_get_extension().your_entrypoint(...).
            """
        ).strip("\n")
        forward_lines = [
            "# Baseline fallback keeps the official PyTorch forward path.",
            "# After implementing CUDA_CPP_SRC / CUDA_CU_SRC you can switch this to _stark_get_extension().your_entrypoint(...).",
        ]
        if baseline_forward_body.strip():
            forward_lines.append(baseline_forward_body.rstrip("\n"))
        else:
            forward_lines.append("raise NotImplementedError('Empty baseline forward body for generic CUDA scaffold')")
        forward_body = "\n".join(forward_lines).rstrip() + "\n"
        return helper_body, cpp_body, cu_body, forward_body

    @staticmethod
    def _extract_grounded_regions(source_code: str) -> list[GroundedRegion]:
        pattern = re.compile(
            r"(?ms)^[ \t]*#\s*<<<IMPROVE:(?P<name>[^>]+)>>>\s*\n(?P<body>.*?)(?=^[ \t]*#\s*<<<END_IMPROVE>>>)"
        )
        regions: list[GroundedRegion] = []
        for match in pattern.finditer(source_code):
            start_line = source_code.count("\n", 0, match.start()) + 1
            end_line = source_code.count("\n", 0, match.end()) + 1
            name = match.group("name")
            excerpt = textwrap.dedent(match.group("body")).strip("\n")
            regions.append(
                GroundedRegion(
                    anchor_name=name,
                    region_role=_region_role(name),
                    start_line=start_line,
                    end_line=end_line,
                    source_excerpt=excerpt,
                    source_hash=hashlib.sha1(excerpt.encode("utf-8")).hexdigest()[:12],
                )
            )
        return regions

    def _build_cases(self, target: dict[str, Any], kind: str) -> list[TestCase]:
        torch = _require_torch()
        case_specs = target.get(f"{kind}_case_specs")
        if case_specs:
            cases: list[TestCase] = []
            default_init_args = list(target["init_args"])
            default_init_kwargs = dict(target.get("init_kwargs", {}))
            for index, spec in enumerate(case_specs, start=1):
                args = [self._materialize_case_value(item) for item in spec.get("args", [])]
                kwargs = {
                    key: self._materialize_case_value(value)
                    for key, value in dict(spec.get("kwargs", {})).items()
                }
                init_args = list(spec.get("init_args", default_init_args))
                init_kwargs = dict(spec.get("init_kwargs", default_init_kwargs))
                cases.append(
                    TestCase(
                        label=f"{kind}-{index}",
                        args=args,
                        kwargs=kwargs,
                        init_args=init_args,
                        init_kwargs=init_kwargs,
                    )
                )
            return cases
        shapes = target[f"{kind}_shapes"]
        init_args = list(target["init_args"])
        init_kwargs = dict(target.get("init_kwargs", {}))
        cases: list[TestCase] = []
        for index, shape in enumerate(shapes, start=1):
            if target["input_kind"] == "symmetric":
                tensor = _build_symmetric_tensor(torch, shape)
            else:
                tensor = torch.rand(shape, dtype=torch.float32)
            cases.append(
                TestCase(
                    label=f"{kind}-{index}",
                    args=[tensor],
                    kwargs={},
                    init_args=list(init_args),
                    init_kwargs=dict(init_kwargs),
                )
            )
        return cases

    @staticmethod
    def _materialize_case_value(spec: Any) -> Any:
        if not isinstance(spec, dict):
            return spec
        torch = _require_torch()
        kind = str(spec.get("kind", "")).strip().lower()
        shape = tuple(spec.get("shape", ()))
        dtype_name = str(spec.get("dtype", "float32")).strip()
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise BridgeLoadError(f"Unsupported torch dtype in case spec: {dtype_name}")
        if kind == "rand":
            return torch.rand(shape, dtype=dtype)
        if kind == "symmetric":
            return _build_symmetric_tensor(torch, shape).to(dtype=dtype)
        if kind == "randint":
            low = int(spec.get("low", 0))
            high = int(spec["high"])
            return torch.randint(low, high, shape, dtype=dtype)
        if kind == "zeros":
            return torch.zeros(shape, dtype=dtype)
        if kind == "ones":
            return torch.ones(shape, dtype=dtype)
        raise BridgeLoadError(f"Unsupported case spec kind: {kind}")

    @staticmethod
    def _strategy_catalog_for_backend(
        level_problem: tuple[int, int],
        backend: str,
        base_catalog: list[StrategySpec],
    ) -> list[StrategySpec]:
        if backend != "cuda":
            return list(base_catalog)
        if level_problem == (1, 25):
            return [
                StrategySpec(
                    name="swish_cuda_forward_call",
                    anchor_name="forward_body",
                    strategy_summary="Keep the Swish path on the native CUDA extension while making the tensor preparation explicit.",
                    instruction="Replace the forward body with a contiguous input temporary followed by the swish_cuda extension call.",
                    expected_gain="Route the activation through the native CUDA kernel while preserving exact Swish math.",
                    good_body="x_contiguous = x.contiguous()\nreturn _stark_get_extension().swish_cuda(x_contiguous)\n",
                    broken_body="return _stark_get_extension().swish_cuda(x) + x\n",
                    debug_body="x_contiguous = x.contiguous()\nreturn _stark_get_extension().swish_cuda(x_contiguous)\n",
                    broken_failure_type="correctness_error",
                )
            ]
        if level_problem == (1, 47):
            return [
                StrategySpec(
                    name="sum_dim1_cuda_forward_call",
                    anchor_name="forward_body",
                    strategy_summary="Keep the reduction on the native CUDA extension while preserving the keepdim contract.",
                    instruction="Replace the forward body with a dim check, a contiguous input temporary, and the sum_dim1_keepdim_cuda extension call.",
                    expected_gain="Route the reduction through the native CUDA kernel without changing the output shape contract.",
                    good_body=(
                        "if self.dim != 1:\n"
                        "    return torch.sum(x, dim=self.dim, keepdim=True)\n"
                        "x_contiguous = x.contiguous()\n"
                        "return _stark_get_extension().sum_dim1_keepdim_cuda(x_contiguous)\n"
                    ),
                    broken_body="return _stark_get_extension().sum_dim1_keepdim_cuda(x).squeeze(1)\n",
                    debug_body=(
                        "if self.dim != 1:\n"
                        "    return torch.sum(x, dim=self.dim, keepdim=True)\n"
                        "x_contiguous = x.contiguous()\n"
                        "return _stark_get_extension().sum_dim1_keepdim_cuda(x_contiguous)\n"
                    ),
                    broken_failure_type="correctness_error",
                )
            ]
        return list(base_catalog)

    @staticmethod
    def _load_python_source(path: Path, function_name: str, label: str) -> str:
        if not path.exists():
            raise BridgeLoadError(f"{label.capitalize()} source file does not exist: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BridgeLoadError(f"Failed to read {label} source file '{path}': {exc}") from exc
        try:
            module = ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno}:{exc.offset}"
            raise BridgeLoadError(f"{label.capitalize()} source has invalid Python syntax at {location}: {exc.msg}") from exc
        if not KernelBenchTaskBridge._has_function(module, function_name):
            raise BridgeLoadError(f"{label.capitalize()} function '{function_name}' was not found in {path}")
        return content

    @staticmethod
    def _has_function(module: ast.AST, function_name: str) -> bool:
        for node in getattr(module, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return True
        return False

    @staticmethod
    def _infer_auto_tags(level: int, stem: str, title: str, backend: str) -> list[str]:
        lowered = f"{stem} {title}".lower()
        tags = ["kernelbench", "official", "gpu", backend, f"level{level}", "auto_bridge"]
        keyword_groups = {
            "matmul": ["matmul", "multiplication", "gemm"],
            "conv": ["conv"],
            "attention": ["attention"],
            "norm": ["norm"],
            "pooling": ["pool", "pooling"],
            "reduction": ["sum", "reduce", "reduction", "cumsum", "scan"],
            "loss": ["loss", "entropy"],
            "activation": ["relu", "sigmoid", "swish", "gelu", "tanh", "softmax", "elu"],
        }
        for tag, keywords in keyword_groups.items():
            if any(keyword in lowered for keyword in keywords):
                tags.append(tag)
        return tags

    @staticmethod
    def _load_runtime_module(problem_path: Path) -> Any:
        module_name = f"_kernelweaver_bridge_{hashlib.sha1(str(problem_path).encode('utf-8')).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, problem_path)
        if spec is None or spec.loader is None:
            raise BridgeLoadError(f"Failed to create an import spec for KernelBench problem: {problem_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _call_runtime_factory(module: Any, function_name: str, problem_path: Path) -> Any:
        factory = getattr(module, function_name, None)
        if not callable(factory):
            raise BridgeLoadError(f"Function '{function_name}' was not found in {problem_path}")
        try:
            return factory()
        except Exception as exc:
            raise BridgeLoadError(f"Failed to execute '{function_name}' from {problem_path}: {exc}") from exc

    @staticmethod
    def _normalize_runtime_values(values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, list):
            return values
        if isinstance(values, tuple):
            return list(values)
        return [values]

    def _reduce_case_arguments(self, args: list[Any], init_args: list[Any], budget: int) -> list[Any]:
        torch = _require_torch()
        reduced_args = list(args)
        tensor_args = [arg for arg in reduced_args if isinstance(arg, torch.Tensor)]
        total_elements = sum(int(tensor.numel()) for tensor in tensor_args)
        if not tensor_args or total_elements <= budget:
            return self._clone_case_values(args)

        protected_scalars = self._collect_scalar_hints(init_args)
        adjustable_dims = []
        shapes: list[list[int] | None] = []
        for arg in reduced_args:
            if not isinstance(arg, torch.Tensor):
                shapes.append(None)
                continue
            current_shape = list(arg.shape)
            shapes.append(current_shape)
            for dim_index, dim_size in enumerate(current_shape):
                if dim_size <= 1:
                    continue
                if dim_size in protected_scalars:
                    continue
                if dim_index == 1 and arg.ndim >= 3 and dim_size <= 256:
                    continue
                if dim_size <= 128:
                    continue
                adjustable_dims.append((len(shapes) - 1, dim_index, dim_size))

        if not adjustable_dims:
            return reduced_args

        factor = min(1.0, (budget / max(total_elements, 1)) ** (1.0 / len(adjustable_dims)))
        for arg_index, dim_index, dim_size in adjustable_dims:
            target_shape = shapes[arg_index]
            if target_shape is None:
                continue
            new_dim = max(1, int(math.floor(dim_size * factor)))
            if new_dim >= dim_size:
                new_dim = dim_size - 1 if dim_size > 1 else dim_size
            target_shape[dim_index] = max(1, new_dim)

        for index, arg in enumerate(reduced_args):
            target_shape = shapes[index]
            if target_shape is None or not isinstance(arg, torch.Tensor):
                continue
            reduced_args[index] = self._slice_tensor(arg, tuple(target_shape))

        self._normalize_target_tensors(reduced_args)
        return reduced_args

    @staticmethod
    def _slice_tensor(tensor: Any, target_shape: tuple[int, ...]) -> Any:
        sliced = tensor
        for dim_index, dim_size in enumerate(target_shape):
            if sliced.shape[dim_index] == dim_size:
                continue
            sliced = sliced.narrow(dim_index, 0, dim_size)
        return sliced.clone()

    @staticmethod
    def _normalize_target_tensors(args: list[Any]) -> None:
        torch = _require_torch()
        float_tensors = [arg for arg in args if isinstance(arg, torch.Tensor) and arg.is_floating_point() and arg.ndim >= 2]
        if not float_tensors:
            return
        for index, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor) or arg.is_floating_point() or arg.ndim == 0:
                continue
            for logits in float_tensors:
                if tuple(arg.shape) != tuple(logits.shape[:-1]):
                    continue
                num_classes = max(1, int(logits.shape[-1]))
                args[index] = torch.remainder(arg, num_classes).to(dtype=arg.dtype)
                break

    def _validate_runtime_case(self, module: Any, init_args: list[Any], args: list[Any]) -> bool:
        torch = _require_torch()
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            return False
        try:
            model = model_cls(*self._clone_case_values(init_args))
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                model(*self._clone_case_values(args))
            return True
        except Exception:
            return False

    @staticmethod
    def _clone_case_values(values: Any) -> Any:
        torch = _require_torch()
        if isinstance(values, torch.Tensor):
            return values.clone()
        if isinstance(values, list):
            return [KernelBenchTaskBridge._clone_case_values(item) for item in values]
        if isinstance(values, tuple):
            return tuple(KernelBenchTaskBridge._clone_case_values(item) for item in values)
        if isinstance(values, dict):
            return {key: KernelBenchTaskBridge._clone_case_values(item) for key, item in values.items()}
        try:
            return copy.deepcopy(values)
        except Exception:
            return values

    @staticmethod
    def _collect_scalar_hints(values: Any) -> set[int]:
        hints: set[int] = set()
        if isinstance(values, bool):
            return hints
        if isinstance(values, int):
            hints.add(int(values))
            return hints
        if isinstance(values, (list, tuple, set)):
            for item in values:
                hints.update(KernelBenchTaskBridge._collect_scalar_hints(item))
            return hints
        if isinstance(values, dict):
            for item in values.values():
                hints.update(KernelBenchTaskBridge._collect_scalar_hints(item))
        return hints


def _is_super_init_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "__init__":
        inner = call.func.value
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "super":
            return True
    return False


def _build_symmetric_tensor(torch_module, shape: tuple[int, ...]):
    total = 1
    for dim in shape:
        total *= dim
    values = torch_module.linspace(-4.0, 4.0, total, dtype=torch_module.float32)
    return values.reshape(shape)


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise BridgeLoadError(f"torch is required for KernelBench task presets: {exc}") from exc
    return torch
