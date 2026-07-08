"""Shared provider error classification utilities."""

from __future__ import annotations


class ProviderTransientError(RuntimeError):
    """A provider/API failure that is safe to retry without consuming a search attempt."""


TRANSIENT_HTTP_STATUS_CODES = frozenset({
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
    520,
    521,
    522,
    523,
    524,
})

TRANSIENT_ERROR_TOKENS = (
    "http 408",
    "http 409",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "http 520",
    "http 521",
    "http 522",
    "http 523",
    "http 524",
    "upstream_error",
    "origin_response_timeout",
    "timeout occurred",
    "temporarily unavailable",
    "timed out",
    "retryable=true",
    '"retryable":true',
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "remote end closed connection",
    "connection reset by peer",
    "connection aborted",
    "connection closed",
    "ssl",
    "response is not valid json",
    "invalid json",
    "stream returned no json events",
    "response content is empty",
    "empty reply from server",
    "chunkedencodingerror",
    "read timed out",
)

PROVIDER_PREFIX_TOKENS = (
    "llm request failed",
    "claude request failed",
    "gemini request failed",
    "provider request failed",
)


def is_retryable_http_status(status_code: int) -> bool:
    return int(status_code) in TRANSIENT_HTTP_STATUS_CODES


def is_transient_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, ProviderTransientError):
        return True
    if not isinstance(exc, (RuntimeError, TimeoutError)):
        return False
    message = str(exc).lower()
    if not message:
        return False
    has_transient_token = any(token in message for token in TRANSIENT_ERROR_TOKENS)
    if isinstance(exc, TimeoutError):
        return has_transient_token and any(prefix in message for prefix in PROVIDER_PREFIX_TOKENS)
    return has_transient_token
