"""Claude-compatible provider using the Anthropic-style /v1/messages API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from ..utils import extract_anchor_names
from .openai_provider import (
    OpenAICompatibleProvider,
    _compact_payload_text,
    _env_or_default,
    _env_override,
    _is_retryable_llm_error,
    _optional_string,
    _retry_delay_seconds,
)


@dataclass
class ClaudeCompatibleConfig:
    """Resolved configuration for a Claude-compatible backend."""

    api_key: str
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-3-5-sonnet-20241022"
    api_version: str = "2023-06-01"
    timeout_seconds: int = 300
    max_tokens: int = 4096
    plan_temperature: float = 0.7
    code_temperature: float = 0.2
    debug_temperature: float = 0.1
    reasoning_effort: str | None = None
    plan_reasoning_effort: str | None = None
    code_reasoning_effort: str | None = None
    debug_reasoning_effort: str | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    user_agent: str = "curl/8.5.0"


class ClaudeCompatibleProvider(OpenAICompatibleProvider):
    """Provider that talks to a Claude-compatible /v1/messages endpoint."""

    name = "claude-compatible"

    def __init__(self, config: ClaudeCompatibleConfig) -> None:
        self.config = config

    def with_overrides(self, **overrides) -> "ClaudeCompatibleProvider":
        return ClaudeCompatibleProvider(replace(self.config, **overrides))

    @classmethod
    def from_env(cls, defaults: dict[str, Any] | None = None) -> "ClaudeCompatibleProvider":
        defaults = defaults or {}
        api_key = str(_env_or_default("CLAUDE_API_KEY", defaults.get("api_key", ""))).strip()
        base_url = str(_env_or_default("CLAUDE_BASE_URL", defaults.get("base_url", "https://api.anthropic.com"))).strip().rstrip("/")
        model = str(_env_or_default("CLAUDE_MODEL", defaults.get("model", "claude-3-5-sonnet-20241022"))).strip()
        api_version = str(_env_or_default("CLAUDE_API_VERSION", defaults.get("api_version", "2023-06-01"))).strip() or "2023-06-01"
        timeout_seconds = int(str(_env_or_default("CLAUDE_TIMEOUT_SECONDS", defaults.get("timeout_seconds", 300))).strip() or "300")
        max_tokens = int(str(_env_or_default("CLAUDE_MAX_TOKENS", defaults.get("max_tokens", 4096))).strip() or "4096")
        plan_temperature = float(str(_env_or_default("CLAUDE_PLAN_TEMPERATURE", defaults.get("plan_temperature", 0.7))).strip() or "0.7")
        code_temperature = float(str(_env_or_default("CLAUDE_CODE_TEMPERATURE", defaults.get("code_temperature", 0.2))).strip() or "0.2")
        debug_temperature = float(str(_env_or_default("CLAUDE_DEBUG_TEMPERATURE", defaults.get("debug_temperature", 0.1))).strip() or "0.1")
        reasoning_effort = _optional_string(_env_or_default("CLAUDE_REASONING_EFFORT", defaults.get("reasoning_effort")))
        plan_reasoning_effort = _env_override(
            "CLAUDE_PLAN_REASONING_EFFORT",
            _optional_string(defaults.get("plan_reasoning_effort")) or reasoning_effort,
        )
        code_reasoning_effort = _env_override(
            "CLAUDE_CODE_REASONING_EFFORT",
            _optional_string(defaults.get("code_reasoning_effort")) or reasoning_effort,
        )
        debug_reasoning_effort = _env_override(
            "CLAUDE_DEBUG_REASONING_EFFORT",
            _optional_string(defaults.get("debug_reasoning_effort")) or reasoning_effort,
        )
        max_retries = max(1, int(str(_env_or_default("CLAUDE_MAX_RETRIES", defaults.get("max_retries", 3))).strip() or "3"))
        retry_backoff_seconds = max(0.0, float(str(_env_or_default("CLAUDE_RETRY_BACKOFF_SECONDS", defaults.get("retry_backoff_seconds", 1.0))).strip() or "1"))
        user_agent = str(_env_or_default("CLAUDE_USER_AGENT", defaults.get("user_agent", "curl/8.5.0"))).strip() or "curl/8.5.0"
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is required for the claude-compatible provider.")
        return cls(
            ClaudeCompatibleConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                api_version=api_version,
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                plan_temperature=plan_temperature,
                code_temperature=code_temperature,
                debug_temperature=debug_temperature,
                reasoning_effort=reasoning_effort,
                plan_reasoning_effort=plan_reasoning_effort,
                code_reasoning_effort=code_reasoning_effort,
                debug_reasoning_effort=debug_reasoning_effort,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                user_agent=user_agent,
            )
        )

    def _chat(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        del reasoning_effort
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            try:
                payload = self._messages_request(system_prompt, user_payload, temperature)
                return self._extract_text_from_messages(payload)
            except TimeoutError as exc:
                last_error = exc
            except RuntimeError as exc:
                last_error = exc
                if not _is_retryable_llm_error(exc):
                    raise
            if attempt + 1 < attempts:
                delay_seconds = _retry_delay_seconds(
                    exc=last_error,
                    attempt_index=attempt,
                    default_backoff=self.config.retry_backoff_seconds,
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Claude request failed before receiving a response")

    def _messages_request(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": _compact_payload_text(user_payload),
                }
            ],
            "stream": False,
            "max_tokens": self.config.max_tokens,
            "temperature": temperature,
        }
        return self._post_json(self._messages_endpoint(), request_body)

    def _messages_endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/messages"

    def _post_json(self, url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "anthropic-version": self.config.api_version,
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Claude request failed: {exc}") from exc

    @staticmethod
    def _extract_text_from_messages(payload: dict[str, Any]) -> str:
        content_text = OpenAICompatibleProvider._content_to_text(payload.get("content"))
        if content_text:
            return content_text
        message_text = OpenAICompatibleProvider._content_to_text((payload.get("message") or {}).get("content"))
        if message_text:
            return message_text
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()
        if payload.get("choices"):
            return OpenAICompatibleProvider._extract_text_from_chat_completions(payload)
        raise RuntimeError(f"Claude response does not contain text content: {payload}")
