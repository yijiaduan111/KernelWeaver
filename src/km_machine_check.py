#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
machine_check_runner.py

- Loads deterministic gating rules from YAML (machine_check layer)
- Normalizes NCU metrics (CSV row) into env fields
- Evaluates predicates / headroom tiers with a strict AST-whitelist evaluator
- Looks up decision_table cases -> allowed_methods (supports case-level gate_when)
- Supports aggregation over multiple rows per kernel
- Computes kernel_launch_count
- Supports optional code feature extraction from CUDA code via:
    (A) heuristic scan: extract_code_features_from_cuda()
    (B) LLM-based code feature extraction: extract_code_features_with_llm()

Security/Correctness:
- No eval/exec
- AST whitelist with strict type checking
- Disallow: string literals, if-expressions (ternary), attribute chains,
  subscripts, comprehensions, lambda, etc.

Notes:
- “code_features” are a set of extra, schema-validated fields defined by
  YAML.machine_check.input_normalization.code_features_schema.
- These fields can be derived by deterministic scans or LLM inference.
  Regardless of source, they MUST be validated/clamped against the YAML schema.
- Nsight Compute (NCU) metrics are treated as the source for numeric counters.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Callable

import yaml


# =============================================================================
# YAML IO
# =============================================================================

def load_yaml_rules(yaml_path: Path) -> dict:
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


# =============================================================================
# Helpers: numeric parsing
# =============================================================================

def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, bool):
        # avoid treating bool as int
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


# =============================================================================
# Expression normalization (YAML -> Python-ish tokens)
# - YAML may use: AND/OR/NOT, true/false, and '=' for boolean equality.
# =============================================================================

_BOOL_EQ_PATTERNS = [
    (re.compile(r"(?<![=!<>])\s=\s(true)\b", re.IGNORECASE), r" == True"),
    (re.compile(r"(?<![=!<>])\s=\s(false)\b", re.IGNORECASE), r" == False"),
]
# Numeric equality: " = 0", " = 1", " = 42", etc. (so YAML "kernel_structure_id = 0" becomes valid Python)
_NUM_EQ_PATTERN = re.compile(r"(?<![=!<>])\s=\s(-?\d+(?:\.\d+)?)\b")


def normalize_expr(expr: str) -> str:
    s = " ".join(str(expr).split())
    s = re.sub(r"\bAND\b", "and", s)
    s = re.sub(r"\bOR\b", "or", s)
    s = re.sub(r"\bNOT\b", "not", s)
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    for pat, rep in _BOOL_EQ_PATTERNS:
        s = pat.sub(rep, s)
    s = _NUM_EQ_PATTERN.sub(r" == \1", s)
    return s


# =============================================================================
# Strict AST evaluator (whitelist + type checks)
# Allowed:
# - numeric constants (int/float), bool constants, None
# - names from env
# - ops: + - * / // % ** (numeric), unary +/-, comparisons, and/or/not
# - calls: max(a,b,...), min(a,b,...), abs(x)
# Disallowed:
# - string literals, ternary (IfExp), subscripts, attributes, comprehensions,
#   lambda, dict/list/set literals, etc.
# - type mixing in arithmetic / boolean ops
# =============================================================================

ALLOWED_FUNC_NAMES = {"max", "min", "abs"}


class SafeEvalError(ValueError):
    pass


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    return type(v).__name__


def _ensure_number(v: Any, ctx: str) -> float:
    if isinstance(v, bool):
        raise SafeEvalError(f"{ctx}: expected number, got bool")
    if isinstance(v, (int, float)):
        return float(v)
    raise SafeEvalError(f"{ctx}: expected number, got {_type_name(v)}")


def _ensure_bool(v: Any, ctx: str) -> bool:
    if isinstance(v, bool):
        return v
    raise SafeEvalError(f"{ctx}: expected bool, got {_type_name(v)}")


def safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    expr = normalize_expr(expr).strip()

    # forbid any string literals early
    if '"' in expr or "'" in expr:
        raise SafeEvalError(f"String literals are forbidden: {expr}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Bad expression syntax after normalization: {expr}") from e

    def eval_node(node: ast.AST) -> Any:
        # constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)) or node.value is None:
                return node.value
            raise SafeEvalError(f"Forbidden literal type: {type(node.value).__name__}")

        # names
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in ALLOWED_FUNC_NAMES:
                return node.id
            raise SafeEvalError(f"Unknown identifier: {node.id}")

        # unary
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                v = eval_node(node.operand)
                return not _ensure_bool(v, "not")
            if isinstance(node.op, (ast.UAdd, ast.USub)):
                v = eval_node(node.operand)
                n = _ensure_number(v, "unary +/-")
                return +n if isinstance(node.op, ast.UAdd) else -n
            raise SafeEvalError("Forbidden unary op")

        # binop (numbers only)
        if isinstance(node, ast.BinOp):
            l = _ensure_number(eval_node(node.left), "binop left")
            r = _ensure_number(eval_node(node.right), "binop right")
            op = node.op
            if isinstance(op, ast.Add):
                return l + r
            if isinstance(op, ast.Sub):
                return l - r
            if isinstance(op, ast.Mult):
                return l * r
            if isinstance(op, ast.Div):
                return l / r
            if isinstance(op, ast.FloorDiv):
                return l // r
            if isinstance(op, ast.Mod):
                return l % r
            if isinstance(op, ast.Pow):
                return l ** r
            raise SafeEvalError("Forbidden binary op")

        # boolop (bool only)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for v in node.values:
                    if not _ensure_bool(eval_node(v), "and"):
                        return False
                return True
            if isinstance(node.op, ast.Or):
                for v in node.values:
                    if _ensure_bool(eval_node(v), "or"):
                        return True
                return False
            raise SafeEvalError("Forbidden bool op")

        # comparisons
        if isinstance(node, ast.Compare):
            left_val = eval_node(node.left)
            cur = left_val
            for op, comp in zip(node.ops, node.comparators):
                right_val = eval_node(comp)

                if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    lnum = _ensure_number(cur, "compare left")
                    rnum = _ensure_number(right_val, "compare right")
                    ok = (
                        lnum < rnum if isinstance(op, ast.Lt) else
                        lnum <= rnum if isinstance(op, ast.LtE) else
                        lnum > rnum if isinstance(op, ast.Gt) else
                        lnum >= rnum
                    )
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    # allow only same-category equality
                    if isinstance(cur, bool) or isinstance(right_val, bool):
                        lb = _ensure_bool(cur, "eq bool left")
                        rb = _ensure_bool(right_val, "eq bool right")
                        ok = (lb == rb) if isinstance(op, ast.Eq) else (lb != rb)
                    else:
                        lnum = _ensure_number(cur, "eq num left")
                        rnum = _ensure_number(right_val, "eq num right")
                        ok = (lnum == rnum) if isinstance(op, ast.Eq) else (lnum != rnum)
                else:
                    raise SafeEvalError("Forbidden comparison op")

                if not ok:
                    return False
                cur = right_val
            return True

        # calls: max/min/abs (numbers only)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in ALLOWED_FUNC_NAMES:
                args = [eval_node(a) for a in node.args]
                if fn.id in ("max", "min"):
                    if len(args) < 1:
                        raise SafeEvalError(f"{fn.id} requires >=1 arg")
                    nums = [_ensure_number(a, f"{fn.id} arg") for a in args]
                    return max(nums) if fn.id == "max" else min(nums)
                if fn.id == "abs":
                    if len(args) != 1:
                        raise SafeEvalError("abs requires exactly 1 arg")
                    return abs(_ensure_number(args[0], "abs arg"))
            raise SafeEvalError("Forbidden function call")

        # forbid ternary explicitly
        if isinstance(node, ast.IfExp):
            raise SafeEvalError("Ternary expressions are forbidden")

        # everything else forbidden
        raise SafeEvalError(f"Forbidden AST node: {type(node).__name__}")

    return eval_node(tree.body)


# =============================================================================
# CSV IO
# =============================================================================

def read_metric_rows(metric_csv_path: Path) -> List[Dict[str, str]]:
    with metric_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _match_kernel_name(r: Dict[str, str], kernel_filter: Optional[str]) -> bool:
    if kernel_filter is None:
        return True
    kn = r.get("Kernel Name") or ""
    return (kn == kernel_filter) or (kernel_filter in kn)


def select_kernel_rows(rows: List[Dict[str, str]], kernel_filter: Optional[str]) -> List[Dict[str, str]]:
    if not rows:
        return []
    if kernel_filter is None:
        return [rows[0]]
    matched = [r for r in rows if _match_kernel_name(r, kernel_filter)]
    return matched if matched else [rows[0]]


def aggregate_kernel_rows(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Aggregate multiple kernel rows into one by taking median for numeric-ish fields."""
    if not rows:
        return {}

    keys = list(rows[0].keys())
    out: Dict[str, str] = dict(rows[0])

    def is_number_str(s: str) -> bool:
        s2 = str(s).strip().replace(",", "")
        if s2 == "" or s2.lower() == "nan":
            return False
        try:
            float(s2)
            return True
        except Exception:
            return False

    for k in keys:
        vals = [r.get(k) for r in rows]
        nums: List[float] = []
        for v in vals:
            if v is not None and is_number_str(v):
                nums.append(float(str(v).strip().replace(",", "")))
        if nums:
            nums.sort()
            mid = len(nums) // 2
            med = nums[mid] if len(nums) % 2 == 1 else 0.5 * (nums[mid - 1] + nums[mid])
            out[k] = str(med)
    return out


# =============================================================================
# Field mapping with fallback for '.avg' mismatches
# =============================================================================

def _get_with_fallback(metric_row: Dict[str, str], raw_key: str) -> Optional[str]:
    if raw_key in metric_row:
        return metric_row.get(raw_key)

    candidates: List[str] = []
    if raw_key.endswith(".avg"):
        candidates.append(raw_key[:-4])
    else:
        candidates.append(raw_key + ".avg")

    for ck in candidates:
        if ck in metric_row:
            return metric_row.get(ck)
    return None


# =============================================================================
# Code features (schema-driven table) - loaded from external file
# =============================================================================

def _load_code_feature_semantics() -> Dict[str, Dict[str, str]]:
    """Load code feature semantics from gate_value_from_kernel_struct file."""
    gate_file = Path(__file__).resolve().parents[1] / "memorybank" / "gate_value_from_kernel_struct"
    
    if not gate_file.exists():
        return {}
    
    semantics: Dict[str, Dict[str, str]] = {}
    
    with open(gate_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header lines (first 2 lines are table header and separator)
    for line in lines[2:]:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        
        # Parse markdown table row: | key | type | range | default | meaning | value semantics | LLM judgment |
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 7:
            continue
        
        key = parts[1].strip('`').strip()
        field_type = parts[2].strip()
        range_val = parts[3].strip()
        default_val = parts[4].strip()
        meaning = parts[5].strip()
        value_semantics = parts[6].strip()
        # parts[7] would be LLM judgment criteria, but we don't need it for semantics dict
        
        if key:
            semantics[key] = {
                "meaning": meaning,
                "value_semantics": value_semantics,
            }
    
    return semantics


def build_code_feature_table(rules: dict) -> str:
    """Return a markdown table describing code_features_schema for LLM/tooling.
    
    Loads semantics from gate_value_from_kernel_struct file and combines with YAML schema.
    """
    schema = rules["machine_check"]["input_normalization"].get("code_features_schema", {})
    semantics = _load_code_feature_semantics()

    headers = ["key", "type", "range", "default", "meaning", "value semantics"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for k, spec in schema.items():
        t = str(spec.get("type", ""))
        lo = spec.get("min", None)
        hi = spec.get("max", None)
        if t == "int" and lo is not None and hi is not None:
            rng = f"{lo}..{hi}"
        elif t == "bool":
            rng = "true|false"
        else:
            rng = ""
        default = spec.get("default", "")

        sem = semantics.get(k, {})
        meaning = sem.get("meaning", "(not defined in gate_value_from_kernel_struct)")
        value_sem = sem.get("value_semantics", "")

        row = [
            k,
            t,
            rng,
            str(default),
            meaning.replace("|", "\\|"),
            value_sem.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# =============================================================================
# Code features: heuristic extractor (LLM extraction moved to judge_gate.py)
# =============================================================================

_KERNEL_STRUCTURE_NAME = {0: "S0", 1: "S1", 2: "S2", 3: "S3", 4: "S4"}


def extract_code_features_from_cuda(cuda_code: str) -> Dict[str, Any]:
    """Lightweight heuristics ONLY (can be wrong). Recommended: manual override tags.

    Supported manual tags:
      // @has_reuse: true
      // @streaming_no_reuse: false
      // @has_vector_load_store: true
      // @is_aligned_vector_access: true
      // @has_tail_handling_overhead: false
      // @kernel_structure_id: 2
      // @has_multiple_kernels_in_forward: true
      // @is_pointwise: true
      // @uses_transcendentals: true
      // @is_naive_gemm: true
      // @has_k_loop: true
      // @is_gemm_kloop: true
      // @is_stencil_conv: true
      // @has_shared_memory_tile: true
      // @uses_vector_types: true
      // @has_bounds_check: true
    """
    s = cuda_code or ""
    s_lower = s.lower()

    def read_tag_bool(tag: str) -> Optional[bool]:
        m = re.search(rf"@{re.escape(tag)}\s*:\s*(true|false)\b", s_lower)
        if not m:
            return None
        return True if m.group(1) == "true" else False

    def read_tag_int(tag: str) -> Optional[int]:
        m = re.search(rf"@{re.escape(tag)}\s*:\s*([0-9]+)\b", s_lower)
        if not m:
            return None
        return int(m.group(1))

    # vector patterns
    vec_patterns = [
        r"\bfloat4\b", r"\bint4\b", r"\bhalf2\b", r"\b__half2\b",
        r"ld\.global\.v\d", r"st\.global\.v\d",
        r"reinterpret_cast<\s*float4\s*\*>", r"reinterpret_cast<\s*int4\s*\*>",
    ]
    has_vec = any(re.search(p, s) for p in vec_patterns)

    # alignment heuristic (very rough)
    # If code uses float4 and also uses pointer casts, assume alignment is intended.
    # Otherwise default to True (optimistic) and let LLM/manual override correct it.
    is_aligned = bool(re.search(r"reinterpret_cast<\s*float4\s*\*>", s) or re.search(r"reinterpret_cast<\s*int4\s*\*>", s))
    is_aligned = True if (has_vec and is_aligned) else True

    # tail handling heuristic
    # Look for patterns like (idx < N), remainder branches, masks.
    has_tail = bool(re.search(r"if\s*\(\s*[^\)]*<\s*[^\)]*\)\s*\{", s_lower) and ("n" in s_lower or "num" in s_lower or "size" in s_lower))

    # reuse heuristic: shared + loop
    has_shared = ("__shared__" in s) or ("extern __shared__" in s)
    shared_access = len(re.findall(r"\bshared\b|\btile\b|\bsmem\b", s_lower))
    loop_like = bool(re.search(r"\bfor\s*\(", s)) or bool(re.search(r"\bwhile\s*\(", s))
    has_reuse = bool(has_shared and (shared_access >= 3) and loop_like)

    # streaming heuristic: default true unless reuse suggests otherwise
    streaming_no_reuse = not has_reuse

    # structure heuristics
    multiple_launches = len(re.findall(r"<<\s*<", s)) >= 2

    reduction_markers = [
        r"__shfl_", r"atomicadd", r"cub::blockreduce",
        r"block_reduce", r"warp_reduce",
    ]
    has_reduction = any(re.search(p, s_lower) for p in reduction_markers)

    irregular_markers = [
        r"indices\s*\[", r"offsets\s*\[", r"scatter", r"gather",
        r"switch\s*\(",
    ]
    has_irregular = any(re.search(p, s_lower) for p in irregular_markers)

    reuse_friendly = bool(has_shared and (shared_access >= 5))

    if multiple_launches:
        ks = 4
    elif has_reduction:
        ks = 3
    elif has_irregular:
        ks = 2
    elif reuse_friendly:
        ks = 1
    else:
        ks = 0

    # TensorCore eligibility heuristic (very rough)
    # Look for GEMM patterns, supported dtypes, and dimension hints
    tc_eligible = False
    if "gemm" in s_lower or "matmul" in s_lower or "matrix" in s_lower:
        # Check for TensorCore-compatible data types
        tc_dtypes = ["half", "__half", "bfloat16", "int8", "int4"]
        has_tc_dtype = any(dtype in s_lower for dtype in tc_dtypes)
        # Check for dimension hints (multiples of 16/32)
        has_dim_hints = bool(re.search(r"\b(16|32|64|128|256)\b", s))
        if has_tc_dtype and has_dim_hints:
            tc_eligible = True

    # NEW: code-structure strong signals (deterministic scan)
    # Pointwise: out[idx] depends only on inp[idx] (no cross-index reads)
    # Look for patterns like: out[i] = f(inp[i]) without reading inp[j] where j != i
    is_pointwise = False
    # Heuristic: if kernel has simple elementwise pattern (no nested loops over different indices)
    # This is a rough heuristic; manual override is recommended
    if not has_reuse and not has_reduction and not has_irregular:
        # Simple elementwise operations are likely pointwise
        is_pointwise = bool(re.search(r"out\s*\[.*\]\s*=\s*.*inp\s*\[.*\]", s_lower) or 
                           re.search(r"out\s*\[.*\]\s*=\s*.*x\s*\[.*\]", s_lower))

    # Transcendental/SFU-heavy math: exp, log, tanh, sin, cos, etc.
    transcendental_patterns = [
        r"\bexp\s*\(", r"\bexpf\s*\(", r"\bexp2\s*\(", r"\bexp2f\s*\(",
        r"\blog\s*\(", r"\blogf\s*\(", r"\blog2\s*\(", r"\blog2f\s*\(", r"\blog10\s*\(", r"\blog10f\s*\(",
        r"\btanh\s*\(", r"\btanhf\s*\(",
        r"\bsin\s*\(", r"\bsinf\s*\(", r"\bcos\s*\(", r"\bcosf\s*\(",
        r"\bsqrt\s*\(", r"\bsqrtf\s*\(", r"\brsqrt\s*\(", r"\brsqrtf\s*\(",
        r"\bpow\s*\(", r"\bpowf\s*\(",
        r"\berf\s*\(", r"\berff\s*\(",
        r"\berfc\s*\(", r"\berfcf\s*\(",
    ]
    uses_transcendentals = any(re.search(p, s) for p in transcendental_patterns)

    # Naive GEMM: explicit K-loop accumulation into out[row,col]
    # Pattern: for(k=0; k<K; k++) { acc += A[row*K+k] * B[k*N+col]; } out[row*N+col] = acc;
    naive_gemm_patterns = [
        r"for\s*\([^)]*k\s*[<>=]",  # K loop
        r"acc\s*\+=\s*.*\[.*\*.*k",  # Accumulation with K dimension
        r"out\s*\[.*row.*\*.*col",  # Output indexed by row and col
    ]
    is_naive_gemm = all(re.search(p, s_lower) for p in naive_gemm_patterns[:2])  # At least K-loop and accumulation

    # K-loop: reduction/accumulation loop (e.g., for(k=0;k<K;k++))
    has_k_loop = bool(re.search(r"for\s*\([^)]*k\s*[<>=]", s_lower) or 
                      re.search(r"for\s*\([^)]*k\s*\+\+", s_lower))
    
    # GEMM K-loop: 2D output, K loop, linear indexing (A[row*K+k] * B[k*N+col]), no spatial window
    # Pattern: acc += A[row*K+k] * B[k*N+col] with 2D output indexing (row, col)
    gemm_kloop_patterns = [
        r"for\s*\([^)]*k\s*[<>=]",  # K loop
        r"acc\s*\+=\s*.*\[.*\*.*k",  # Accumulation with K dimension
        r"\[.*row.*\*.*col",  # 2D output indexing (row, col)
        r"\[.*\*.*k.*\]",  # Linear indexing with K dimension
    ]
    # Check for GEMM-like patterns but exclude stencil/conv patterns (no kd/kh/kw)
    has_gemm_kloop = all(re.search(p, s_lower) for p in gemm_kloop_patterns[:2])  # At least K-loop and accumulation
    # Exclude if has spatial window dimensions (kd, kh, kw)
    if has_gemm_kloop and (re.search(r"\bkd\b|\bkh\b|\bkw\b", s_lower) or 
                           re.search(r"\[.*od.*oh.*ow", s_lower) or
                           re.search(r"\[.*id.*ih.*iw", s_lower)):
        has_gemm_kloop = False
    
    # Stencil/Conv: sliding window access pattern (od,oh,ow) + (kd,kh,kw) with accumulation
    # Pattern: nested loops over spatial dimensions (od/oh/ow) and kernel dimensions (kd/kh/kw)
    # with accumulation to a single output point
    stencil_conv_patterns = [
        r"for\s*\([^)]*od\s*[<>=]",  # Output depth loop
        r"for\s*\([^)]*oh\s*[<>=]",  # Output height loop
        r"for\s*\([^)]*ow\s*[<>=]",  # Output width loop
        r"for\s*\([^)]*kd\s*[<>=]",  # Kernel depth loop
        r"for\s*\([^)]*kh\s*[<>=]",  # Kernel height loop
        r"for\s*\([^)]*kw\s*[<>=]",  # Kernel width loop
    ]
    # Check for stencil/conv patterns: at least one output spatial dim (od/oh/ow) AND one kernel dim (kd/kh/kw)
    has_output_spatial = any(re.search(p, s_lower) for p in stencil_conv_patterns[:3])
    has_kernel_spatial = any(re.search(p, s_lower) for p in stencil_conv_patterns[3:])
    # Also check for accumulation pattern (acc += or val +=)
    has_accumulation = bool(re.search(r"acc\s*\+=", s_lower) or re.search(r"val\s*\+=", s_lower) or
                           re.search(r"\+\s*=", s_lower))
    is_stencil_conv = bool(has_output_spatial and has_kernel_spatial and has_accumulation)

    # Shared memory tiling: explicit __shared__ or extern shared
    has_shared_memory_tile = bool(has_shared and (shared_access >= 3))

    # Vector types: float2/float4/half2 or vectorized casts
    vector_type_patterns = [
        r"\bfloat2\b", r"\bfloat4\b", r"\bhalf2\b", r"\b__half2\b",
        r"\bint2\b", r"\bint4\b", r"\buint2\b", r"\buint4\b",
        r"reinterpret_cast<\s*float[24]\s*\*>", r"reinterpret_cast<\s*half2\s*\*>",
    ]
    uses_vector_types = any(re.search(p, s) for p in vector_type_patterns) or has_vec

    # Bounds checks: if(idx<N) / if(row<B && col<OUT)
    # Default to true (most kernels have bounds checks)
    bounds_check_patterns = [
        r"if\s*\([^)]*<\s*[^)]*\)",  # if (idx < N)
        r"if\s*\([^)]*>\s*[^)]*\)",  # if (idx > N)
        r"if\s*\([^)]*<=\s*[^)]*\)",  # if (idx <= N)
        r"if\s*\([^)]*>=\s*[^)]*\)",  # if (idx >= N)
    ]
    has_bounds_check = any(re.search(p, s_lower) for p in bounds_check_patterns) if s else True  # Default True

    # CUDA Graph eligibility heuristic (very rough, conservative default)
    # Check for patterns that suggest fixed shapes and stable control flow
    # This is a conservative heuristic; manual override or LLM judgment is recommended
    # Default to False (conservative) - requires explicit indicators to be True
    cudagraph_eligible = False
    
    # Look for indicators of fixed shapes and stable control flow:
    # - No dynamic shape operations (variable batch size, dynamic padding)
    # - No runtime-dependent branches based on tensor values
    # - No dynamic allocations in forward path
    # - Inference mode patterns (eval, no training)
    
    # Negative indicators (disqualify CUDA Graph)
    dynamic_shape_markers = [
        r"\.shape\[", r"\.size\(\)", r"\.numel\(\)",  # Shape queries (may indicate dynamic)
        r"torch\.empty\([^)]*\[",  # Dynamic allocation with variable size
        r"\.view\([^)]*\)", r"\.reshape\([^)]*\)",  # Reshape operations (may be dynamic)
    ]
    has_dynamic_shapes = any(re.search(p, s_lower) for p in dynamic_shape_markers)
    
    # Look for runtime-dependent control flow (branches based on tensor values)
    runtime_branch_markers = [
        r"if\s*\([^)]*\[.*\]",  # Branches based on tensor indexing
        r"switch\s*\(",  # Switch statements (may be runtime-dependent)
    ]
    has_runtime_branches = any(re.search(p, s_lower) for p in runtime_branch_markers)
    
    # Look for training mode indicators (dropout, batch norm training, etc.)
    training_markers = [
        r"self\.training", r"\.training\s*=", r"dropout", r"batch_norm.*training",
    ]
    has_training_mode = any(re.search(p, s_lower) for p in training_markers)
    
    # Positive indicators (suggest CUDA Graph eligibility)
    # Only set to True if we have positive indicators AND no negative indicators
    if not has_dynamic_shapes and not has_runtime_branches and not has_training_mode:
        # Look for fixed-size patterns or inference mode
        inference_markers = [
            r"\.eval\(\)", r"eval\(\)", r"inference", r"torch\.no_grad",
            r"cudagraph", r"cuda.*graph",  # Explicit CUDA Graph usage
        ]
        has_inference_mode = any(re.search(p, s_lower) for p in inference_markers)
        # Conservative: only set True if we have explicit inference mode indicators
        if has_inference_mode:
            cudagraph_eligible = True

    # manual overrides (most reliable)
    overrides_bool = {
        "has_reuse": read_tag_bool("has_reuse"),
        "streaming_no_reuse": read_tag_bool("streaming_no_reuse"),
        "has_vector_load_store": read_tag_bool("has_vector_load_store"),
        "is_aligned_vector_access": read_tag_bool("is_aligned_vector_access"),
        "has_tail_handling_overhead": read_tag_bool("has_tail_handling_overhead"),
        "has_multiple_kernels_in_forward": read_tag_bool("has_multiple_kernels_in_forward"),
        "cudagraph_eligible": read_tag_bool("cudagraph_eligible"),
        "tc_eligible": read_tag_bool("tc_eligible"),
        "is_pointwise": read_tag_bool("is_pointwise"),
        "uses_transcendentals": read_tag_bool("uses_transcendentals"),
        "is_naive_gemm": read_tag_bool("is_naive_gemm"),
        "has_k_loop": read_tag_bool("has_k_loop"),
        "is_gemm_kloop": read_tag_bool("is_gemm_kloop"),
        "is_stencil_conv": read_tag_bool("is_stencil_conv"),
        "has_shared_memory_tile": read_tag_bool("has_shared_memory_tile"),
        "uses_vector_types": read_tag_bool("uses_vector_types"),
        "has_bounds_check": read_tag_bool("has_bounds_check"),
    }
    for k, v in overrides_bool.items():
        if v is not None:
            if k == "has_reuse":
                has_reuse = v
            elif k == "streaming_no_reuse":
                streaming_no_reuse = v
            elif k == "has_vector_load_store":
                has_vec = v
            elif k == "is_aligned_vector_access":
                is_aligned = v
            elif k == "has_tail_handling_overhead":
                has_tail = v
            elif k == "has_multiple_kernels_in_forward":
                multiple_launches = v
            elif k == "cudagraph_eligible":
                cudagraph_eligible = v
            elif k == "tc_eligible":
                tc_eligible = v
            elif k == "is_pointwise":
                is_pointwise = v
            elif k == "uses_transcendentals":
                uses_transcendentals = v
            elif k == "is_naive_gemm":
                is_naive_gemm = v
            elif k == "has_k_loop":
                has_k_loop = v
            elif k == "is_gemm_kloop":
                has_gemm_kloop = v
            elif k == "is_stencil_conv":
                is_stencil_conv = v
            elif k == "has_shared_memory_tile":
                has_shared_memory_tile = v
            elif k == "uses_vector_types":
                uses_vector_types = v
            elif k == "has_bounds_check":
                has_bounds_check = v

    tag_ks = read_tag_int("kernel_structure_id")
    if tag_ks is not None:
        ks = tag_ks

    return {
        "has_reuse": bool(has_reuse),
        "streaming_no_reuse": bool(streaming_no_reuse),
        "has_vector_load_store": bool(has_vec),
        "is_aligned_vector_access": bool(is_aligned),
        "has_tail_handling_overhead": bool(has_tail),
        "kernel_structure_id": int(ks),
        "has_multiple_kernels_in_forward": bool(multiple_launches),
        "cudagraph_eligible": bool(cudagraph_eligible),
        "tc_eligible": bool(tc_eligible),
        "is_pointwise": bool(is_pointwise),
        "uses_transcendentals": bool(uses_transcendentals),
        "is_naive_gemm": bool(is_naive_gemm),
        "has_k_loop": bool(has_k_loop),
        "is_gemm_kloop": bool(has_gemm_kloop),
        "is_stencil_conv": bool(is_stencil_conv),
        "has_shared_memory_tile": bool(has_shared_memory_tile),
        "uses_vector_types": bool(uses_vector_types),
        "has_bounds_check": bool(has_bounds_check),
    }


def validate_code_features_against_schema(rules: dict, code_features: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clamp values against YAML code_features_schema."""
    schema = rules["machine_check"]["input_normalization"].get("code_features_schema", {})
    out: Dict[str, Any] = dict(code_features)

    for k, spec in schema.items():
        t = spec.get("type", "")
        default = spec.get("default", None)
        v = out.get(k, default)

        if t == "bool":
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                # avoid numeric -> bool auto cast; accept only {0,1}
                out[k] = bool(v) if v in (0, 1) else bool(default)
            elif isinstance(v, str):
                vl = v.strip().lower()
                if vl in ("true", "false"):
                    out[k] = (vl == "true")
                else:
                    out[k] = bool(default)
            else:
                out[k] = bool(default)

        elif t == "int":
            lo = spec.get("min", None)
            hi = spec.get("max", None)
            try:
                if isinstance(v, bool):
                    raise ValueError("bool is not int")
                vi = int(v)
            except Exception:
                vi = int(default) if default is not None else 0
            if lo is not None and vi < int(lo):
                vi = int(lo)
            if hi is not None and vi > int(hi):
                vi = int(hi)
            out[k] = vi

        else:
            # unknown type => keep default
            out[k] = default

    # keep only schema keys (drop extras)
    out = {k: out.get(k, schema[k].get("default")) for k in schema.keys()}
    return out


# =============================================================================
# Core: compute_fields, predicates, tiering, lookup (with gate_when)
# =============================================================================


def compute_fields(
    rules: dict,
    metric_row: Dict[str, str],
    *,
    code_features: Optional[dict] = None,
    kernel_launch_count: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    mc = rules["machine_check"]
    norm = mc["input_normalization"]
    fm = norm["field_mapping"]
    derived = norm.get("derived_fields", {})
    schema = norm.get("code_features_schema", {})

    env: Dict[str, Any] = {}
    missing: List[str] = []

    code_features = code_features or {}

    # load code features defaults
    for k, spec in schema.items():
        env[k] = code_features.get(k, spec.get("default"))

    # kernel_launch_count available to YAML if used
    if kernel_launch_count is not None:
        env["kernel_launch_count"] = int(kernel_launch_count)
    else:
        env.setdefault("kernel_launch_count", 0)

    # kernel_structure string derived from kernel_structure_id
    ksid = env.get("kernel_structure_id", 0)
    if isinstance(ksid, (int, float)) and not isinstance(ksid, bool):
        ksid_int = int(ksid)
    else:
        ksid_int = 0
    env["kernel_structure"] = _KERNEL_STRUCTURE_NAME.get(ksid_int, "S0")

    # mapped metric fields
    for k, raw_key in fm.items():
        raw_val = _get_with_fallback(metric_row, raw_key)
        if raw_val is None:
            missing.append(f"{k} <- {raw_key}")
            env[k] = float("nan")
        else:
            env[k] = _to_float(raw_val)

    # derived fields
    for k, expr in derived.items():
        expr_s = normalize_expr(str(expr))
        try:
            env[k] = safe_eval(expr_s, env)
        except Exception as e:
            raise SafeEvalError(f"Failed derived field {k} = {expr_s}: {e}") from e

    return env, missing


def match_headroom_tier(rules: dict, env: Dict[str, Any]) -> str:
    tiers = rules["machine_check"]["headroom_tiers"]
    order = ["Tier-H", "Tier-M", "Tier-L"]
    for tier_name in order:
        if tier_name not in tiers:
            continue
        rule_expr = normalize_expr(str(tiers[tier_name]["rule"]))
        try:
            if bool(safe_eval(rule_expr, env)):
                return tier_name
        except SafeEvalError:
            continue
    return "Tier-M"


def predicate_true(pred_expr: Any, env: Dict[str, Any]) -> bool:
    if isinstance(pred_expr, str):
        return bool(safe_eval(pred_expr, env))
    if isinstance(pred_expr, dict):
        if "any" in pred_expr:
            return any(predicate_true(x, env) for x in pred_expr["any"])
        if "all" in pred_expr:
            return all(predicate_true(x, env) for x in pred_expr["all"])
    raise SafeEvalError(f"Bad predicate expr: {pred_expr}")


def match_signatures(rules: dict, env: Dict[str, Any]) -> List[str]:
    predlib = rules["machine_check"]["ncu_predicates"]
    matched: List[str] = []
    for name, expr in predlib.items():
        try:
            if predicate_true(expr, env):
                matched.append(name)
        except SafeEvalError:
            continue
    return matched


def apply_priority_rules(rules: dict, env: Dict[str, Any]) -> Optional[str]:
    prules = rules["machine_check"].get("bottleneck_priority_rules", [])
    for r in prules:
        when_expr = normalize_expr(str(r["when"]))
        try:
            if bool(safe_eval(when_expr, env)):
                return r["force_bottleneck"]
        except SafeEvalError:
            continue
    return None


def _gate_when_ok(case: Dict[str, Any], env: Dict[str, Any]) -> bool:
    if "gate_when" not in case or case["gate_when"] in (None, ""):
        return True
    expr = normalize_expr(str(case["gate_when"]))
    try:
        return bool(safe_eval(expr, env))
    except SafeEvalError:
        return False


def lookup_case(
    rules: dict,
    forced_bottleneck: Optional[str],
    tier: str,
    kernel_structure: str,
    matched_preds: List[str],
    env: Dict[str, Any],
) -> Dict[str, Any]:
    dt = rules["machine_check"]["decision_table"]
    bottlenecks = [forced_bottleneck] if forced_bottleneck else list(dt.keys())

    # Separate bottlenecks with and without ncu_signature
    bottlenecks_with_sig = []
    fallback_bottlenecks = []
    
    for b in bottlenecks:
        if b not in dt:
            continue
        entry = dt[b]
        sig = entry.get("ncu_signature", [])
        sig_any = entry.get("ncu_signature_any", [])
        
        if sig or sig_any:
            bottlenecks_with_sig.append(b)
        else:
            # No signature specified: treat as fallback bottleneck
            fallback_bottlenecks.append(b)
    
    # First, check bottlenecks with ncu_signature
    for b in bottlenecks_with_sig:
        entry = dt[b]
        sig = entry.get("ncu_signature", [])
        sig_any = entry.get("ncu_signature_any", [])
        
        # Check if signature matches
        sig_matches = False
        if sig_any:
            # ncu_signature_any: match if ANY of the signature lists fully matches (all predicates in that list)
            # Each signature list in sig_any must be completely matched (all predicates satisfied)
            sig_matches = any(all(s in matched_preds for s in sig_list) for sig_list in sig_any)
        elif sig:
            # ncu_signature: match if ALL predicates in the signature list match
            sig_matches = all(s in matched_preds for s in sig)
        
        if not sig_matches:
            continue

        for c in entry.get("cases", []):
            headroom = c["headroom"]
            headroom_ok = (tier in headroom) if isinstance(headroom, list) else (tier == headroom)
            if not headroom_ok:
                continue

            ks = c["kernel_structure"]
            ks_ok = (kernel_structure in ks) if isinstance(ks, list) else (kernel_structure == ks)
            if not ks_ok:
                continue

            if not _gate_when_ok(c, env):
                continue

            return {
                "bottleneck_id": b,
                "case_id": c.get("id", "UNKNOWN_CASE"),
                "allowed_methods": c.get("allowed_methods", []),
                "forbidden_methods": c.get("forbidden_methods", []),
                "expected_speedup": c.get("expected_speedup", ""),
                "requirements": c.get("requirements", {}),
                "gate_when": c.get("gate_when", ""),
            }
    
    # If no bottleneck with signature matched, check fallback bottlenecks (no ncu_signature required)
    for b in fallback_bottlenecks:
        entry = dt[b]
        
        for c in entry.get("cases", []):
            headroom = c["headroom"]
            headroom_ok = (tier in headroom) if isinstance(headroom, list) else (tier == headroom)
            if not headroom_ok:
                continue

            ks = c["kernel_structure"]
            ks_ok = (kernel_structure in ks) if isinstance(ks, list) else (kernel_structure == ks)
            if not ks_ok:
                continue

            if not _gate_when_ok(c, env):
                continue

            return {
                "bottleneck_id": b,
                "case_id": c.get("id", "UNKNOWN_CASE"),
                "allowed_methods": c.get("allowed_methods", []),
                "forbidden_methods": c.get("forbidden_methods", []),
                "expected_speedup": c.get("expected_speedup", ""),
                "requirements": c.get("requirements", {}),
                "gate_when": c.get("gate_when", ""),
            }

    return {
        "bottleneck_id": forced_bottleneck or "unknown",
        "case_id": "NO_MATCH",
        "allowed_methods": [],
        "forbidden_methods": [],
        "expected_speedup": "",
        "requirements": {},
        "gate_when": "",
    }


# =============================================================================
# Public entry
# =============================================================================


def run_machine_check(
    yaml_path: Path,
    metric_csv_path: Path,
    *,
    kernel_filter: Optional[str] = None,
    cuda_code: str = "",
    arch_path: Optional[Path] = None,  # PyTorch reference path for judge_gate
    feature_mode: str = "heuristic",  # heuristic|llm|manual
    code_features: Optional[dict] = None,
    call_llm: Optional[Callable] = None,  # LLM call function for judge_gate (signature: (prompt, sys_prompt, ...) -> str)
    aggregate: bool = True,
    kernel_launch_count: Optional[int] = None,  # Optional kernel launch count from nsys (overrides len(rows))
    io_dir: Optional[Path] = None,  # Optional directory to save judge_gate prompts and replies
    round_idx: Optional[int] = None,  # Optional round index for filename
) -> Dict[str, Any]:
    """Run machine-check and return a diagnostic JSON-like dict.
    
    Args:
        yaml_path: Path to machine_check YAML rules
        metric_csv_path: Path to NCU metrics CSV file
        kernel_filter: Optional kernel name filter
        cuda_code: Optional CUDA kernel code string
        arch_path: Optional PyTorch reference architecture path (for judge_gate LLM extraction)
        feature_mode: "heuristic" (default), "llm" (use judge_gate), or "manual" (code_features provided)
        code_features: Optional pre-extracted code features (used when feature_mode="manual")
        call_llm: Optional LLM call function for judge_gate extraction (used when feature_mode="llm")
        aggregate: Whether to aggregate multiple kernel rows
        kernel_launch_count: Optional kernel launch count from nsys (overrides len(rows) calculation)
    
    Returns:
        Dict containing tier, bottleneck_id, case_id, allowed_methods, code_features_used, etc.
    """
    rules = load_yaml_rules(yaml_path)

    rows = read_metric_rows(metric_csv_path)
    if not rows:
        raise ValueError(f"No rows in CSV: {metric_csv_path}")

    # Determine kernel_launch_count: prioritize parameter (from nsys), fall back to len(rows) (from NCU)
    kernel_launch_count_final = kernel_launch_count if kernel_launch_count is not None else len(rows)

    krows = select_kernel_rows(rows, kernel_filter)
    metric_row = aggregate_kernel_rows(krows) if aggregate else krows[0]

    # code_features source
    if feature_mode == "manual":
        cf = validate_code_features_against_schema(rules, code_features or {})
    elif feature_mode == "llm":
        # Use judge_gate.py to extract code_features via LLM
        if call_llm is None or not cuda_code:
            # Fall back to heuristic if call_llm or cuda_code not available
            cf = extract_code_features_from_cuda(cuda_code) if cuda_code else {}
            cf = validate_code_features_against_schema(rules, cf)
        else:
            # Import judge_gate module
            try:
                from prompts.judge_gate import build_gate_prompts
                
                # Build prompts using judge_gate
                if arch_path is None:
                    # Try to infer arch_path from cuda_code context (may not always work)
                    # For now, use a dummy path - caller should provide arch_path
                    import warnings
                    warnings.warn("arch_path not provided for judge_gate, using heuristic fallback")
                    cf = extract_code_features_from_cuda(cuda_code) if cuda_code else {}
                    cf = validate_code_features_against_schema(rules, cf)
                else:
                    gate_sys_prompt, gate_instruction = build_gate_prompts(
                        arch_path=arch_path,
                        cuda_code=cuda_code,
                    )
                    
                    # Save judge_gate prompt if io_dir and round_idx are provided
                    if io_dir is not None and round_idx is not None:
                        try:
                            io_dir.mkdir(parents=True, exist_ok=True)
                            gate_prompt_file = io_dir / f"round{round_idx:03d}_judge_gate_prompt.txt"
                            gate_prompt_content = f"=== SYSTEM PROMPT ===\n{gate_sys_prompt}\n\n=== INSTRUCTION ===\n{gate_instruction}"
                            gate_prompt_file.write_text(gate_prompt_content, encoding="utf-8")
                            print(f"[judge_gate] Saved prompt to: {gate_prompt_file}")
                        except Exception as e:
                            print(f"[judge_gate] Warning: Failed to save prompt: {e}")
                    
                    # Call LLM to extract code features
                    gate_raw = call_llm(
                        gate_instruction,
                        gate_sys_prompt,
                        log_path=None,
                        call_type="judge_gate",
                        round_idx=round_idx if round_idx is not None else -1,
                    )
                    
                    # Save judge_gate reply if io_dir and round_idx are provided
                    if io_dir is not None and round_idx is not None:
                        try:
                            io_dir.mkdir(parents=True, exist_ok=True)
                            gate_reply_file = io_dir / f"round{round_idx:03d}_judge_gate_reply.txt"
                            gate_reply_file.write_text(gate_raw, encoding="utf-8")
                            print(f"[judge_gate] Saved reply to: {gate_reply_file}")
                        except Exception as e:
                            print(f"[judge_gate] Warning: Failed to save reply: {e}")
                    
                    # Parse JSON response
                    try:
                        # Extract JSON from response (handle markdown code blocks)
                        gate_raw_clean = gate_raw.strip()
                        if "```json" in gate_raw_clean:
                            gate_raw_clean = gate_raw_clean.split("```json")[1].split("```")[0].strip()
                        elif "```" in gate_raw_clean:
                            gate_raw_clean = gate_raw_clean.split("```")[1].split("```")[0].strip()
                        
                        cf = json.loads(gate_raw_clean)
                        cf = validate_code_features_against_schema(rules, cf)
                        
                        # Save extracted code_features JSON if io_dir and round_idx are provided
                        if io_dir is not None and round_idx is not None:
                            try:
                                io_dir.mkdir(parents=True, exist_ok=True)
                                code_features_file = io_dir / f"round{round_idx:03d}_code_features.json"
                                with open(code_features_file, 'w', encoding='utf-8') as f:
                                    json.dump(cf, f, indent=2, ensure_ascii=False)
                                print(f"[judge_gate] Saved code_features to: {code_features_file}")
                            except Exception as e:
                                print(f"[judge_gate] Warning: Failed to save code_features: {e}")
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Failed to parse code_features from judge_gate LLM response: {e}. Using heuristic fallback.")
                        cf = extract_code_features_from_cuda(cuda_code) if cuda_code else {}
                        cf = validate_code_features_against_schema(rules, cf)
            except ImportError:
                import warnings
                warnings.warn("judge_gate module not found, using heuristic fallback")
                cf = extract_code_features_from_cuda(cuda_code) if cuda_code else {}
                cf = validate_code_features_against_schema(rules, cf)
    else:
        # heuristic mode (default)
        cf = extract_code_features_from_cuda(cuda_code) if cuda_code else {}
        cf = validate_code_features_against_schema(rules, cf)

    env, missing = compute_fields(
        rules,
        metric_row,
        code_features=cf,
        kernel_launch_count=kernel_launch_count_final,
    )

    tier = match_headroom_tier(rules, env)
    matched_preds = match_signatures(rules, env)
    forced = apply_priority_rules(rules, env)
    kernel_structure = env.get("kernel_structure", "S0")

    case = lookup_case(rules, forced, tier, kernel_structure, matched_preds, env)

    key_metrics = {
        "dram_throughput_pct": env.get("dram_throughput_pct"),
        "l2_throughput_pct": env.get("l2_throughput_pct"),
        "l1_throughput_pct": env.get("l1_throughput_pct"),
        "sm_throughput_pct": env.get("sm_throughput_pct"),
        "achieved_occupancy_pct": env.get("achieved_occupancy_pct"),
        "registers_per_thread": env.get("registers_per_thread"),
        "memory_dependency_stall_pct": env.get("memory_dependency_stall_pct"),
        "primary_limiter_util_pct": env.get("primary_limiter_util_pct"),
        "kernel_duration_us": env.get("kernel_duration_us"),
        "kernel_launch_count": env.get("kernel_launch_count"),
        "branch_divergence_pct": env.get("branch_divergence_pct"),
        "warp_execution_efficiency_pct": env.get("warp_execution_efficiency_pct"),
    }

    return {
        "tier": tier,
        "kernel_structure": kernel_structure,
        "kernel_structure_id": env.get("kernel_structure_id", 0),
        "feature_mode": feature_mode,
        "code_features_used": {k: env.get(k) for k in rules["machine_check"]["input_normalization"].get("code_features_schema", {}).keys()},
        "matched_predicates": matched_preds,
        "forced_bottleneck": forced,
        **case,
        "key_metrics": key_metrics,
        "missing_fields": missing,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, required=True, help="Path to machine_check YAML")
    ap.add_argument("--csv", type=str, required=True, help="Path to Nsight Compute export CSV")
    ap.add_argument("--kernel", type=str, default=None, help="Kernel name filter (substring or exact)")
    ap.add_argument("--cuda_code", type=str, default="", help="Optional CUDA code path for feature extraction")
    ap.add_argument("--feature_mode", type=str, default="heuristic", choices=["heuristic", "llm", "manual"])
    ap.add_argument("--code_features_json", type=str, default="", help="Manual code_features JSON when feature_mode=manual")
    ap.add_argument("--no_aggregate", action="store_true", help="Disable aggregation (use first matching row)")

    ap.add_argument("--print_code_feature_table", action="store_true", help="Print code_features table from YAML and exit")

    args = ap.parse_args()

    rules = load_yaml_rules(Path(args.yaml))

    cuda_code_str = ""
    if args.cuda_code:
        cuda_code_str = Path(args.cuda_code).read_text(encoding="utf-8")

    if args.print_code_feature_table:
        print(build_code_feature_table(rules))
        raise SystemExit(0)

    manual_cf: Optional[dict] = None
    if args.feature_mode == "manual":
        if not args.code_features_json:
            raise SystemExit("--feature_mode=manual requires --code_features_json")
        manual_cf = json.loads(args.code_features_json)

    if args.feature_mode == "llm":
        raise SystemExit(
            "feature_mode=llm is deprecated. Use judge_gate.py to extract code_features "
            "and pass them with feature_mode=manual."
        )

    out = run_machine_check(
        Path(args.yaml),
        Path(args.csv),
        kernel_filter=args.kernel,
        cuda_code=cuda_code_str,
        feature_mode=args.feature_mode,
        code_features=manual_cf,
        call_llm=None,  # CLI doesn't support LLM calls
        aggregate=(not args.no_aggregate),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
