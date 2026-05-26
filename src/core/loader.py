"""Thin KernelBench loader used by the STARK workflow.

This module deliberately avoids per-problem adapter specs. It only reads an
official KernelBench problem, builds a generic ModelNew scaffold, and leaves
optimization target selection to the Plan Agent.
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backends import is_cute_backend, is_native_cuda_backend, is_supported_kernelbench_backend, is_tilelang_backend, supported_kernelbench_backends_text
from ..models import GroundedRegion, TaskSpec, TestCase
from ..semantics import SemanticAnalyzer, SemanticProfile


class KernelBenchLoadError(ValueError):
    pass


@dataclass
class ProblemInfo:
    path: Path
    source_code: str
    description: str
    imports_block: str
    class_preamble: str
    init_signature: str
    init_body: str
    forward_signature: str
    forward_body: str
    forward_steps: list[str]


class KernelBenchLoader:
    """Load official KernelBench problems without handwritten task adapters."""

    def load_official_problem(
        self,
        kernelbench_root: str | Path,
        level: int,
        problem_id: int,
        backend: str = "triton",
        semantics_enabled: bool = True,
        semantics_mode: str = "rule",
        semantics_max_anchor_hints: int = 6,
    ) -> TaskSpec:
        if not is_supported_kernelbench_backend(backend):
            raise KernelBenchLoadError(
                f"Unsupported KernelBench backend: {backend}. Supported backends: {supported_kernelbench_backends_text()}."
            )
        root = Path(kernelbench_root)
        problem_path = self._resolve_problem_path(root, level, problem_id)
        info = self._inspect_problem(problem_path)
        scaffold = self._build_scaffold(info, level=level, problem_id=problem_id, backend=backend)
        grounded_regions = self._extract_grounded_regions(scaffold)
        semantic_profile = self._build_semantic_profile(
            info,
            grounded_regions,
            backend=backend,
            enabled=semantics_enabled,
            mode=semantics_mode,
            max_anchor_hints=semantics_max_anchor_hints,
        )
        return TaskSpec(
            name=self._task_name(level, problem_id, problem_path),
            description=f"KernelBench Level {level} / Problem {problem_id}: {info.description}",
            source_code=scaffold,
            reference_code=info.source_code,
            function_name="ModelNew",
            reference_function_name="Model",
            test_cases=[],
            benchmark_cases=[],
            tags=self._tags(level, backend, problem_path),
            strategy_catalog=[],
            source_origin=str(problem_path),
            benchmark_family="kernelbench",
            entry_kind="model_class",
            level=level,
            problem_id=problem_id,
            backend=backend,
            source_root=str(root),
            grounded_regions=grounded_regions,
            semantic_profile=semantic_profile,
        )


    @staticmethod
    def _build_semantic_profile(
        info: ProblemInfo,
        grounded_regions: list[GroundedRegion],
        backend: str,
        enabled: bool,
        mode: str,
        max_anchor_hints: int,
    ) -> SemanticProfile | None:
        if not enabled:
            return SemanticProfile(enabled=False, mode=mode, op_type="disabled", summary="Semantic analysis is disabled.")
        return SemanticAnalyzer().analyze(
            info,
            grounded_regions,
            backend=backend,
            mode=mode,
            max_anchor_hints=max_anchor_hints,
        )

    def _resolve_problem_path(self, kernelbench_root: Path, level: int, problem_id: int) -> Path:
        candidate_dirs = [
            kernelbench_root / "KernelBench" / f"level{level}",
            kernelbench_root / f"level{level}",
        ]
        existing_dirs = [path for path in candidate_dirs if path.exists()]
        if not existing_dirs:
            locations = ", ".join(str(path) for path in candidate_dirs)
            raise KernelBenchLoadError(f"KernelBench level directory does not exist. Checked: {locations}")
        for level_dir in existing_dirs:
            matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
            if matches:
                return matches[0]
        searched = ", ".join(str(path) for path in existing_dirs)
        raise KernelBenchLoadError(f"KernelBench problem L{level}/P{problem_id} was not found under: {searched}")

    def _inspect_problem(self, path: Path) -> ProblemInfo:
        source = path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno}:{exc.offset}"
            raise KernelBenchLoadError(f"Official KernelBench source has invalid syntax at {location}: {exc.msg}") from exc
        model_class = self._find_class(module, "Model", path)
        self._find_function(module, "get_inputs", path)
        self._find_function(module, "get_init_inputs", path)
        init_node = self._find_method(model_class, "__init__", path)
        forward_node = self._find_method(model_class, "forward", path)
        return ProblemInfo(
            path=path,
            source_code=source,
            description=ast.get_docstring(model_class) or path.stem.replace("_", " "),
            imports_block=self._extract_imports(source, module),
            class_preamble=self._extract_class_preamble(source, model_class),
            init_signature=self._extract_signature(source, init_node),
            init_body=self._extract_body(source, init_node, drop_super_init=True),
            forward_signature=self._extract_signature(source, forward_node),
            forward_body=self._extract_body(source, forward_node, drop_super_init=False),
            forward_steps=self._extract_forward_steps(source, forward_node),
        )

    def _build_scaffold(self, info: ProblemInfo, level: int, problem_id: int, backend: str) -> str:
        if is_native_cuda_backend(backend):
            return self._build_cuda_scaffold(info, level, problem_id)
        if is_tilelang_backend(backend):
            return self._build_dsl_scaffold(
                info,
                level=level,
                helper_lines=[
                    "_STARK_TILELANG_KERNEL = None",
                    "",
                    "def _stark_import_tilelang():",
                    "    import tilelang",
                    "    import tilelang.language as T",
                    "    return tilelang, T",
                ],
                kernel_anchor="tilelang_kernel",
                kernel_comment="Define or cache TileLang kernels/helpers here.",
            )
        if is_cute_backend(backend):
            return self._build_dsl_scaffold(
                info,
                level=level,
                helper_lines=[
                    "_STARK_CUTE_KERNEL = None",
                    "",
                    "def _stark_import_cute():",
                    "    import cutlass",
                    "    import cutlass.cute as cute",
                    "    return cutlass, cute",
                ],
                kernel_anchor="cute_kernel",
                kernel_comment="Define or cache CuTe DSL kernels/helpers here.",
            )
        return self._build_python_scaffold(info, level=level)

    def _build_python_scaffold(self, info: ProblemInfo, level: int) -> str:
        parts = self._imports_and_helpers(info.imports_block, helper_lines=[])
        parts.append("class ModelNew(nn.Module):")
        self._append_class_body(parts, info, level=level, forward_intro=[])
        return "\n".join(parts).rstrip() + "\n"

    def _build_dsl_scaffold(
        self,
        info: ProblemInfo,
        level: int,
        helper_lines: list[str],
        kernel_anchor: str,
        kernel_comment: str,
    ) -> str:
        parts = self._imports_and_helpers(info.imports_block, helper_lines=helper_lines)
        parts.extend([
            f"# <<<IMPROVE:{kernel_anchor}>>>",
            f"# {kernel_comment}",
            "# Keep the public ModelNew interface unchanged.",
            "# <<<END_IMPROVE>>>",
            "",
            "class ModelNew(nn.Module):",
        ])
        self._append_class_body(
            parts,
            info,
            level=level,
            forward_intro=["# Baseline fallback keeps the official PyTorch forward path."],
        )
        return "\n".join(parts).rstrip() + "\n"

    def _build_cuda_scaffold(self, info: ProblemInfo, level: int, problem_id: int) -> str:
        imports = self._ensure_cuda_imports(info.imports_block)
        protected_helper_lines = [
            "_STARK_EXTENSION = None",
            "",
            "def _stark_strip_anchor_markers(source: str) -> str:",
            "    cleaned_lines = []",
            "    for line in source.splitlines():",
            "        stripped = line.lstrip()",
            "        if stripped.startswith('# <<<IMPROVE:') or stripped.startswith('# <<<END_IMPROVE>>>'):",
            "            continue",
            "        cleaned_lines.append(line)",
            "    return '\\n'.join(cleaned_lines)",
            "",
            "def _stark_extension_name() -> str:",
            "    digest = hashlib.sha1((_stark_strip_anchor_markers(CUDA_CPP_SRC) + _stark_strip_anchor_markers(CUDA_CU_SRC)).encode('utf-8')).hexdigest()[:12]",
            f"    return f'stark_cuda_l{level}_p{problem_id}_{{digest}}'",
            "",
            "def _stark_get_extension():",
            "    global _STARK_EXTENSION",
            "    if _STARK_EXTENSION is None:",
            "        _STARK_EXTENSION = load_inline(",
            "            name=_stark_extension_name(),",
            "            cpp_sources=_stark_strip_anchor_markers(CUDA_CPP_SRC),",
            "            cuda_sources=_stark_strip_anchor_markers(CUDA_CU_SRC),",
            "            functions=None,",
            "            extra_cflags=['-O3'],",
            "            extra_cuda_cflags=['-O3', '--use_fast_math'],",
            "            with_cuda=True,",
            "            verbose=False,",
            "        )",
            "    return _STARK_EXTENSION",
        ]
        parts: list[str] = []
        imports = imports.strip()
        if imports:
            parts.append(imports)
            parts.append("")
        parts.extend(protected_helper_lines)
        parts.extend([
            "",
            "# <<<IMPROVE:user_helpers>>>",
            "# Add optional pure Python helpers here. Do not edit framework CUDA loader helpers.",
            "# <<<END_IMPROVE>>>",
            "",
        ])
        parts.extend([
            'CUDA_CPP_SRC = r"""',
            "# <<<IMPROVE:cuda_cpp>>>",
            "#include <torch/extension.h>",
            "",
            "// Add pybind exports for custom CUDA entrypoints here.",
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}",
            "# <<<END_IMPROVE>>>",
            '"""',
            "",
            'CUDA_CU_SRC = r"""',
            "# <<<IMPROVE:cuda_cu>>>",
            "#include <torch/extension.h>",
            "#include <cuda.h>",
            "#include <cuda_runtime.h>",
            "",
            "// Add CUDA kernels and exported wrapper functions here.",
            "# <<<END_IMPROVE>>>",
            '"""',
            "",
            "class ModelNew(nn.Module):",
        ])
        self._append_class_body(
            parts,
            info,
            level=level,
            forward_intro=[
                "# Baseline fallback keeps the official PyTorch forward path.",
                "# After implementing CUDA_CPP_SRC / CUDA_CU_SRC, call _stark_get_extension().your_entrypoint(...).",
            ],
        )
        return "\n".join(parts).rstrip() + "\n"

    def _imports_and_helpers(self, imports: str, helper_lines: list[str]) -> list[str]:
        parts: list[str] = []
        imports = imports.strip()
        if imports:
            parts.append(imports)
            parts.append("")
        parts.append("# <<<IMPROVE:helpers>>>")
        parts.extend(helper_lines)
        parts.append("# <<<END_IMPROVE>>>")
        parts.append("")
        return parts

    def _append_class_body(self, parts: list[str], info: ProblemInfo, level: int, forward_intro: list[str]) -> None:
        if info.class_preamble:
            for line in info.class_preamble.splitlines():
                parts.append(f"    {line}" if line else "")
        parts.extend([
            f"    def {info.init_signature}:",
            "        super().__init__()",
            "        # <<<IMPROVE:init_body>>>",
        ])
        if info.init_body:
            for line in info.init_body.splitlines():
                parts.append(f"        {line}" if line else "")
        parts.extend(["        # <<<END_IMPROVE>>>", "", f"    def {info.forward_signature}:"])
        if info.forward_steps:
            for index, step in enumerate(info.forward_steps, start=1):
                parts.append(f"        # <<<IMPROVE:forward_stmt_{index}>>>")
                if index == 1:
                    for intro in forward_intro:
                        parts.append(f"        {intro}")
                if step:
                    for line in step.splitlines():
                        parts.append(f"        {line}" if line else "")
                parts.append("        # <<<END_IMPROVE>>>")
        else:
            parts.append("        # <<<IMPROVE:forward_body>>>")
            for intro in forward_intro:
                parts.append(f"        {intro}")
            if info.forward_body:
                for line in info.forward_body.splitlines():
                    parts.append(f"        {line}" if line else "")
            parts.append("        # <<<END_IMPROVE>>>")
        parts.append("")

    @staticmethod
    def _ensure_cuda_imports(imports: str) -> str:
        lines = [line for line in imports.splitlines() if line.strip()]
        required = ["import hashlib", "from torch.utils.cpp_extension import load_inline"]
        existing = {line.strip() for line in lines}
        for line in required:
            if line not in existing:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _find_class(module: ast.Module, class_name: str, path: Path) -> ast.ClassDef:
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        raise KernelBenchLoadError(f"Class '{class_name}' was not found in {path}")

    @staticmethod
    def _find_function(module: ast.Module, function_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return node
        raise KernelBenchLoadError(f"Function '{function_name}' was not found in {path}")

    @staticmethod
    def _find_method(class_node: ast.ClassDef, method_name: str, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        raise KernelBenchLoadError(f"Method 'Model.{method_name}' was not found in {path}")

    @staticmethod
    def _extract_imports(source: str, module: ast.Module) -> str:
        segments = []
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segments.append(ast.get_source_segment(source, node) or "")
        return "\n".join(segment for segment in segments if segment)

    @staticmethod
    def _extract_class_preamble(source: str, class_node: ast.ClassDef) -> str:
        segments = []
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            segment = ast.get_source_segment(source, node)
            if segment:
                segments.append(textwrap.dedent(segment))
        return "\n".join(segments).strip("\n")

    @staticmethod
    def _extract_signature(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        segment = ast.get_source_segment(source, node)
        if not segment:
            raise KernelBenchLoadError(f"Could not extract signature for {node.name}")
        first_line = segment.splitlines()[0].strip()
        if first_line.endswith(":"):
            first_line = first_line[:-1]
        return first_line[4:].strip() if first_line.startswith('def ') else first_line

    @staticmethod
    def _extract_body(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef, drop_super_init: bool) -> str:
        lines = []
        for statement in node.body:
            if drop_super_init and KernelBenchLoader._is_super_init(statement):
                continue
            segment = ast.get_source_segment(source, statement)
            if segment:
                lines.append(textwrap.dedent(segment))
        return "\n".join(lines).strip("\n")

    @staticmethod
    def _is_super_init(statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Expr):
            return False
        call = statement.value
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        )

    @staticmethod
    def _extract_forward_steps(source: str, forward_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        steps = []
        for statement in forward_node.body:
            segment = ast.get_source_segment(source, statement)
            if segment:
                steps.append(textwrap.dedent(segment).strip("\n"))
        return steps

    @staticmethod
    def _extract_grounded_regions(source_code: str) -> list[GroundedRegion]:
        pattern = re.compile(r"(?ms)^[ \t]*#\s*<<<IMPROVE:(?P<name>[^>]+)>>>\s*\n(?P<body>.*?)(?=^[ \t]*#\s*<<<END_IMPROVE>>>)")
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

    @staticmethod
    def _task_name(level: int, problem_id: int, path: Path) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
        return f"kernelbench_l{level}_{problem_id}_{slug}"

    @staticmethod
    def _tags(level: int, backend: str, path: Path) -> list[str]:
        stem = path.stem.lower()
        tags = ["kernelbench", "official", "gpu", backend, f"level{level}"]
        for token in ("matmul", "conv", "pool", "norm", "reduction", "attention", "loss", "activation"):
            if token in stem:
                tags.append(token)
        return tags


def _region_role(anchor_name: str) -> str:
    if anchor_name == "helpers" or anchor_name.endswith("_kernel") or anchor_name.startswith("cuda_"):
        return "helper"
    if anchor_name == "init_body":
        return "init"
    if anchor_name == "forward_body" or anchor_name.startswith("forward_stmt_"):
        return "forward"
    return "unknown"
