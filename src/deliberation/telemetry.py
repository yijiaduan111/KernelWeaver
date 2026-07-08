"""Structured telemetry for multi-model deliberation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliberationEvent:
    phase: str
    provider_name: str
    status: str
    elapsed_seconds: float | None = None
    detail: str | None = None
