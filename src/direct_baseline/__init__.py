"""Direct LLM baseline for KernelBench tasks.

This package intentionally stays outside the STARK search loop. It asks one
model for one complete KernelBench candidate and evaluates that candidate with
the same paper evaluator used by the main workflow.
"""

from .code_extract import extract_python_code
from .reporting import direct_result_row
from .runner import run_direct_baseline

__all__ = ["direct_result_row", "extract_python_code", "run_direct_baseline"]
