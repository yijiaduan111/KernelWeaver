"""Shared HTTP helpers for provider implementations."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any


def post_json_request(
    *,
    url: str,
    request_body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    error_prefix: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                snippet = raw[:1000]
                raise RuntimeError(f"{error_prefix}: invalid JSON from {url}: {snippet}") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix}: HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"{error_prefix}: timed out after {timeout_seconds}s: {url}") from exc
    except socket.timeout as exc:
        raise TimeoutError(f"{error_prefix}: timed out after {timeout_seconds}s: {url}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise TimeoutError(f"{error_prefix}: timed out after {timeout_seconds}s: {url}") from exc
        raise RuntimeError(f"{error_prefix}: {exc}") from exc
    except (ConnectionError, http.client.HTTPException, ssl.SSLError) as exc:
        raise RuntimeError(f"{error_prefix}: {exc}") from exc
