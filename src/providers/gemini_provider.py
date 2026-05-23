"""Gemini-compatible provider using the Gemini generateContent API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from .openai_provider import (
    OpenAICompatibleProvider,
    _compact_payload_text,
    _env_or_default,
    _env_override,
    _is_retryable_llm_error,
    _optional_string,
)


@dataclass
class GeminiCompatibleConfig:
    """Resolved configuration for a Gemini-compatible backend."""

    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com"
    model: str = "gemini-3.1-pro-preview"
    api_version: str = "v1beta"
    timeout_seconds: int = 300
    max_output_tokens: int = 4096
    plan_temperature: float = 0.7
    code_temperature: float = 0.2
    debug_temperature: float = 0.1
    thinking_include_thoughts: bool = False
    reasoning_effort: str | None = None
    plan_reasoning_effort: str | None = None
    code_reasoning_effort: str | None = None
    debug_reasoning_effort: str | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    user_agent: str = "curl/8.5.0"


class GeminiCompatibleProvider(OpenAICompatibleProvider):
    """Provider that talks to a Gemini-compatible generateContent endpoint."""

    name = "gemini-compatible"

    def __init__(self, config: GeminiCompatibleConfig) -> None:
        self.config = config

    def with_overrides(self, **overrides) -> "GeminiCompatibleProvider":
        return GeminiCompatibleProvider(replace(self.config, **overrides))

    @classmethod
    def from_env(cls, defaults: dict[str, Any] | None = None) -> "GeminiCompatibleProvider":
        defaults = defaults or {}
        api_key = str(_env_or_default("GEMINI_API_KEY", defaults.get("api_key", ""))).strip()
        base_url = str(_env_or_default("GEMINI_BASE_URL", defaults.get("base_url", "https://generativelanguage.googleapis.com"))).strip().rstrip("/")
        model = str(_env_or_default("GEMINI_MODEL", defaults.get("model", "gemini-3.1-pro-preview"))).strip()
        api_version = str(_env_or_default("GEMINI_API_VERSION", defaults.get("api_version", "v1beta"))).strip() or "v1beta"
        timeout_seconds = int(str(_env_or_default("GEMINI_TIMEOUT_SECONDS", defaults.get("timeout_seconds", 300))).strip() or "300")
        max_output_tokens = int(str(_env_or_default("GEMINI_MAX_OUTPUT_TOKENS", defaults.get("max_output_tokens", 4096))).strip() or "4096")
        plan_temperature = float(str(_env_or_default("GEMINI_PLAN_TEMPERATURE", defaults.get("plan_temperature", 0.7))).strip() or "0.7")
        code_temperature = float(str(_env_or_default("GEMINI_CODE_TEMPERATURE", defaults.get("code_temperature", 0.2))).strip() or "0.2")
        debug_temperature = float(str(_env_or_default("GEMINI_DEBUG_TEMPERATURE", defaults.get("debug_temperature", 0.1))).strip() or "0.1")
        thinking_include_thoughts = _env_bool(
            _env_or_default("GEMINI_THINKING_INCLUDE_THOUGHTS", defaults.get("thinking_include_thoughts", False)),
            default=bool(defaults.get("thinking_include_thoughts", False)),
        )
        reasoning_effort = _optional_string(_env_or_default("GEMINI_REASONING_EFFORT", defaults.get("reasoning_effort")))
        plan_reasoning_effort = _env_override(
            "GEMINI_PLAN_REASONING_EFFORT",
            _optional_string(defaults.get("plan_reasoning_effort")) or reasoning_effort,
        )
        code_reasoning_effort = _env_override(
            "GEMINI_CODE_REASONING_EFFORT",
            _optional_string(defaults.get("code_reasoning_effort")) or reasoning_effort,
        )
        debug_reasoning_effort = _env_override(
            "GEMINI_DEBUG_REASONING_EFFORT",
            _optional_string(defaults.get("debug_reasoning_effort")) or reasoning_effort,
        )
        max_retries = max(1, int(str(_env_or_default("GEMINI_MAX_RETRIES", defaults.get("max_retries", 3))).strip() or "3"))
        retry_backoff_seconds = max(0.0, float(str(_env_or_default("GEMINI_RETRY_BACKOFF_SECONDS", defaults.get("retry_backoff_seconds", 1.0))).strip() or "1"))
        user_agent = str(_env_or_default("GEMINI_USER_AGENT", defaults.get("user_agent", "curl/8.5.0"))).strip() or "curl/8.5.0"
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the gemini-compatible provider.")
        return cls(
            GeminiCompatibleConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                api_version=api_version,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                plan_temperature=plan_temperature,
                code_temperature=code_temperature,
                debug_temperature=debug_temperature,
                thinking_include_thoughts=thinking_include_thoughts,
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
                payload = self._generate_content_request(system_prompt, user_payload, temperature)
                return self._extract_text_from_generate_content(payload)
            except TimeoutError as exc:
                last_error = exc
            except RuntimeError as exc:
                last_error = exc
                if not _is_retryable_llm_error(exc):
                    raise
            if attempt + 1 < attempts and self.config.retry_backoff_seconds > 0:
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed before receiving a response")

    def _generate_content_request(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": _compact_payload_text(user_payload)}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        if self.config.thinking_include_thoughts:
            request_body["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}
        return self._post_json(self._generate_content_endpoint(), request_body)

    def _generate_content_endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        version = self.config.api_version.strip("/")
        if not base.endswith(f"/{version}"):
            base = f"{base}/{version}"
        model = urllib.parse.quote(self.config.model, safe="")
        return f"{base}/models/{model}:generateContent?key={urllib.parse.quote(self.config.api_key, safe='')}"

    def _post_json(self, url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

    @staticmethod
    def _extract_text_from_generate_content(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        text_parts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip() and not part.get("thought"):
                    text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)
        if isinstance(payload.get("text"), str) and payload["text"].strip():
            return payload["text"].strip()
        raise RuntimeError(f"Gemini response does not contain answer text: {payload}")


def _env_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default
