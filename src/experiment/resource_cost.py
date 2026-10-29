"""Opt-in HTTP resource accounting, without prompts, responses or credentials.

PYRAMID_COST_LOG enables append-only JSONL records. Instrument the transport,
not just successful agent returns, so retries and rejected requests are counted.
Missing usage is unknown, never zero. This module does not estimate prices or
pretend HTTP time is end-to-end task time.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
import uuid


def normalize_usage(payload: dict, *, protocol: str = "openai") -> dict:
    """Normalize OpenAI-compatible Chat/Responses usage; exclude double counts."""
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    def count(value):
        return value if type(value) is int and value >= 0 else None

    inputs = count(usage.get("prompt_tokens", usage.get("input_tokens")))
    outputs = count(usage.get("completion_tokens", usage.get("output_tokens")))
    uncached = inputs
    cache_read = count(usage.get("cache_read_input_tokens"))
    cache_write = count(usage.get("cache_creation_input_tokens"))
    if protocol == "anthropic" and inputs is not None:
        # Anthropic input_tokens excludes cache reads and writes. Cache fields
        # are optional when caching was not used, unlike missing base usage.
        inputs += (cache_read or 0) + (cache_write or 0)
    total = count(usage.get("total_tokens"))
    if total is None and inputs is not None and outputs is not None:
        total = inputs + outputs
    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details")) or {}
    cached = count(details.get("cached_tokens")) if isinstance(details, dict) else None
    return {"input_tokens": inputs, "output_tokens": outputs,
            "total_tokens": total, "cached_input_tokens": cache_read if protocol == "anthropic" else cached,
            "uncached_input_tokens": uncached if protocol == "anthropic" else None,
            "cache_write_input_tokens": cache_write}


def _append(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def http_attempt(model: str):
    """Each transport invocation has a start and end, including failures.

    A process killed mid-request leaves a detectable unfinished start record.
    Logging errors fail loudly in an explicitly metered run, avoiding silent
    undercounting. Default unmetered experiments are unaffected.
    """
    destination = os.environ.get("PYRAMID_COST_LOG")
    info = {}
    if not destination:
        yield info
        return
    path = Path(destination)
    identity = {"schema_version": 1, "request_id": uuid.uuid4().hex,
                "method": os.environ.get("PYRAMID_COST_METHOD"),
                "task": os.environ.get("PYRAMID_COST_TASK"), "model": model,
                "logical_call_id": os.environ.get("PYRAMID_COST_CALL_ID"),
                "role": os.environ.get("PYRAMID_COST_ROLE")}
    _append(path, {**identity, "event": "http_start",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    status = "response_received"
    try:
        yield info
    except BaseException as exc:
        status = "error"
        info["error_type"] = type(exc).__name__
        raise
    finally:
        allowed = {key: value for key, value in info.items()
                   if key in {"input_tokens", "output_tokens", "total_tokens",
                              "cached_input_tokens", "error_type"}}
        allowed.update({key: value for key, value in info.items()
                        if key in {"uncached_input_tokens", "cache_write_input_tokens"}})
        allowed.update({key: value for key, value in info.items()
                        if key in {"stop_reason", "content_block_types", "text_chars", "text_nonspace_chars"}})
        _append(path, {**identity, **allowed, "event": "http_end", "status": status,
                       "http_seconds": time.perf_counter() - started})


def summarize_http(events: list[dict]) -> dict:
    starts = {event["request_id"]: event for event in events if event.get("event") == "http_start"}
    ends = {event["request_id"]: event for event in events if event.get("event") == "http_end"}
    if not set(ends).issubset(starts):
        raise ValueError("HTTP log contains end records without matching starts")
    result = {"llm_requests_total": len(starts),
              "llm_http_responses_received": sum(e.get("status") == "response_received" for e in ends.values()),
              "llm_http_errors": sum(e.get("status") == "error" for e in ends.values()),
              "llm_requests_unfinished": len(set(starts) - set(ends)),
              "http_seconds_completed_requests": sum(e["http_seconds"] for e in ends.values()),
              "api_cost_estimate": None}
    calls = [e.get("logical_call_id") for e in starts.values()]
    result["llm_logical_calls"] = len(set(calls)) if all(calls) else None
    result["llm_retries"] = len(calls) - len(set(calls)) if all(calls) else None
    for metric in ("input_tokens", "output_tokens", "total_tokens"):
        values = [ends.get(request_id, {}).get(metric) for request_id in starts]
        known = [value for value in values if type(value) is int and value >= 0]
        result[metric + "_known_subtotal"] = sum(known)
        result[metric + "_missing_requests"] = len(values) - len(known)
        result[metric] = sum(known) if len(known) == len(values) else None
    return result


@contextmanager
def logical_call(role: str):
    """Group HTTP retries within a serial logical generation call."""
    keys = {"PYRAMID_COST_CALL_ID": uuid.uuid4().hex, "PYRAMID_COST_ROLE": role}
    old = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def phase_span(kind: str):
    destination = os.environ.get("PYRAMID_COST_LOG")
    started = time.perf_counter()
    identity = {"event": "phase_start", "phase": kind, "span_id": uuid.uuid4().hex,
                "method": os.environ.get("PYRAMID_COST_METHOD"),
                "task": os.environ.get("PYRAMID_COST_TASK")}
    if destination:
        _append(Path(destination), identity)
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if destination:
            _append(Path(destination), {**identity, "event": "phase_end", "status": status,
                                       "seconds": time.perf_counter() - started})
