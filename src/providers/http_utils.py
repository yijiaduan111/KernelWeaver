"""Shared HTTP helpers for provider implementations."""

from __future__ import annotations

import json
from typing import Any

import requests

from .errors import ProviderTransientError, is_retryable_http_status


def post_json_request(
    *,
    url: str,
    request_body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    error_prefix: str,
) -> dict[str, Any]:
    connect_timeout = _connect_timeout_seconds(timeout_seconds)
    read_timeout = max(1.0, float(timeout_seconds))
    body = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        response = requests.post(
            url,
            data=body,
            headers=headers,
            timeout=(connect_timeout, read_timeout),
        )
    except requests.exceptions.Timeout as exc:
        raise ProviderTransientError(
            f"{error_prefix}: timed out after connect={connect_timeout:.1f}s/read={read_timeout:.1f}s: {url}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderTransientError(f"{error_prefix}: {exc}") from exc
    _raise_for_status(response, error_prefix)
    try:
        return response.json()
    except ValueError as exc:
        snippet = response.text[:1000]
        raise RuntimeError(f"{error_prefix}: invalid JSON from {url}: {snippet}") from exc


def stream_json_events_request(
    *,
    url: str,
    request_body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    error_prefix: str,
) -> list[dict[str, Any]]:
    """Post JSON and collect JSON objects from an SSE/line-delimited stream."""
    connect_timeout = _connect_timeout_seconds(timeout_seconds)
    read_timeout = max(1.0, float(timeout_seconds))
    body = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    stream_headers = dict(headers)
    stream_headers.setdefault("Accept", "text/event-stream")
    try:
        response = requests.post(
            url,
            data=body,
            headers=stream_headers,
            timeout=(connect_timeout, read_timeout),
            stream=True,
        )
    except requests.exceptions.Timeout as exc:
        raise ProviderTransientError(
            f"{error_prefix}: stream timed out after connect={connect_timeout:.1f}s/read={read_timeout:.1f}s: {url}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderTransientError(f"{error_prefix}: stream request failed: {exc}") from exc
    _raise_for_status(response, error_prefix)

    events: list[dict[str, Any]] = []
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            line = _normalize_stream_line(raw_line)
            if not line or line.startswith(":") or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data = line[len("data:") :].strip()
            elif line.startswith("{"):
                data = line
            else:
                continue
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    break
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    except requests.exceptions.Timeout as exc:
        raise ProviderTransientError(f"{error_prefix}: stream timed out while reading: {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderTransientError(f"{error_prefix}: stream interrupted while reading: {exc}") from exc

    if not events:
        raise RuntimeError(f"{error_prefix}: stream returned no JSON events from {url}")
    return events


def _raise_for_status(response, error_prefix: str) -> None:
    if response.status_code < 400:
        return
    message = f"{error_prefix}: HTTP {response.status_code}: {response.text}"
    if is_retryable_http_status(response.status_code):
        raise ProviderTransientError(message)
    raise RuntimeError(message)


def _normalize_stream_line(raw_line: Any) -> str:
    if raw_line is None:
        return ""
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8", errors="ignore").strip()
    return str(raw_line).strip()


def _connect_timeout_seconds(timeout_seconds: int) -> float:
    return min(20.0, max(1.0, float(timeout_seconds)))
