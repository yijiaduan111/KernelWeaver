from __future__ import annotations

"""Backend-specific static contract checks for generated candidates."""

from dataclasses import dataclass, field

from ...backends import normalize_backend


@dataclass(frozen=True)
class ContractCheckResult:
    ok: bool
    failure_type: str | None = None
    logs: list[str] = field(default_factory=list)


def pass_result(*logs: str) -> ContractCheckResult:
    return ContractCheckResult(True, logs=[log for log in logs if log])


def fail_result(failure_type: str, message: str) -> ContractCheckResult:
    return ContractCheckResult(False, failure_type, [message])


def check_backend_contract(source_code: str, backend: str | None = None) -> ContractCheckResult:
    """Dispatch to backend-specific checks while keeping non-target DSLs isolated."""
    normalized = normalize_backend(backend, default="")
    if normalized == "cuda":
        from .cuda import check_cuda_contract

        return check_cuda_contract(source_code)
    if _looks_like_cuda_extension(source_code):
        return pass_result(f"backend_contract_skipped_for_backend:{normalized or 'unknown'}")
    return pass_result()


def _looks_like_cuda_extension(source_code: str) -> bool:
    return (
        "CUDA_CPP_SRC" in source_code
        or "CUDA_CU_SRC" in source_code
        or "_stark_get_extension" in source_code
        or "PYBIND11_MODULE" in source_code
    )
