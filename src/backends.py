"""Shared backend names and helpers for KernelBench-style tasks."""

from __future__ import annotations

KERNELBENCH_BACKENDS: tuple[str, ...] = ("triton", "cuda", "tilelang", "cute")


def normalize_backend(name: str | None, default: str = "triton") -> str:
    normalized = str(name or default).strip().lower()
    return normalized or default


def is_supported_kernelbench_backend(name: str | None) -> bool:
    return normalize_backend(name) in KERNELBENCH_BACKENDS


def is_native_cuda_backend(name: str | None) -> bool:
    return normalize_backend(name) == "cuda"


def is_tilelang_backend(name: str | None) -> bool:
    return normalize_backend(name) == "tilelang"


def is_cute_backend(name: str | None) -> bool:
    return normalize_backend(name) == "cute"


def supported_kernelbench_backends_text() -> str:
    return ", ".join(KERNELBENCH_BACKENDS)
