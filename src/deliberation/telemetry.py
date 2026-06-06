"""Structured telemetry for multi-model deliberation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DeliberationEvent:
    phase: Literal["propose", "review"]
    provider_name: str
    status: Literal["start", "ok", "error"]
    elapsed_seconds: float | None = None
    detail: str | None = None
