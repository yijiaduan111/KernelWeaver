"""Claude-compatible provider using the Anthropic-style /v1/messages API."""

from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from ..utils import extract_anchor_names
from ..experiment.resource_cost import http_attempt, logical_call, normalize_usage
from .openai_provider import (
    OpenAICompatibleProvider,
    _compact_payload_text,
    _env_or_default,
    _env_override,
    _is_retryable_llm_error,
    _optional_string,
)


class ClaudeTextResponseError(RuntimeError):
    """A completed, metered response violated this text-only adapter's contract."""


def _response_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Diagnose empty replies from counts only; response content is never logged."""
    content = payload.get("content")
    blocks = [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []
    text = "".join(item["text"] for item in blocks if isinstance(item.get("text"), str))
    return {"stop_reason": payload.get("stop_reason"),
            "content_block_types": [str(item.get("type")) for item in blocks],
            "text_chars": len(text), "text_nonspace_chars": len("".join(text.split()))}


@dataclass(slots=True)
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
        with logical_call(__import__("os").environ.get("PYRAMID_COST_ROLE", "agent")):
            return self._chat_with_retries(system_prompt, user_payload, temperature, reasoning_effort)

    def _chat_with_retries(
        self, system_prompt: str, user_payload: dict[str, Any], temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        del reasoning_effort
        if os.environ.get("CLAUDE_TOOL_RECOVERY", "").lower() in {"1", "true", "yes"}:
            return self._chat_with_denied_tools(system_prompt, user_payload, temperature)
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
            if attempt + 1 < attempts and self.config.retry_backoff_seconds > 0:
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Claude request failed before receiving a response")

    def _chat_with_denied_tools(self, system_prompt, user_payload, temperature):
        """Opt-in transport repair, not a tool executor or an extra agent.

        A gateway may ignore tool_choice=none. Return explicit tool errors so
        the model can finish its text response. Every continuation consumes the
        EXISTING HTTP retry allowance and is metered by _post_json.
        """
        messages = None
        last_error = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            try:
                payload = self._messages_request(system_prompt, user_payload, temperature, messages=messages)
                try:
                    return self._extract_text_from_messages(payload)
                except ClaudeTextResponseError:
                    messages = self._denied_tool_history(payload, messages, user_payload)
                    raise
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
        raise RuntimeError("Claude request failed before receiving a response")

    @staticmethod
    def _denied_tool_history(payload, messages, user_payload):
        content = payload.get("content")
        if not isinstance(content, list):
            return None
        tools = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]
        ids = [item.get("id") for item in tools]
        if (not tools or any(not isinstance(identity, str) or not identity for identity in ids)
                or len(ids) != len(set(ids)) or any(
                    isinstance(item, dict) and item.get("type") == "server_tool_use" for item in content)):
            # Malformed or server-side tool replies are not safe to continue.
            return None
        history = list(messages) if messages is not None else [
            {"role": "user", "content": _compact_payload_text(user_payload)}]
        history.append({"role": "assistant", "content": content})
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": identity, "is_error": True,
             "content": "Tool execution is unavailable. This call was NOT executed. "
                        "Use only the task source and evidence already supplied in the request. "
                        "Return the requested final JSON or code as text; do not call tools."}
            for identity in ids
        ]})
        return history

    def _messages_request(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": messages if messages is not None else [
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
        with http_attempt(str(request_body.get("model", self.config.model))) as event:
            payload = self._post_json_unmetered(url, request_body)
            event.update(normalize_usage(payload, protocol="anthropic"))
            event.update(_response_shape(payload))
            return payload

    def _post_json_unmetered(self, url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        request_body = dict(request_body)
        # These agents consume text, not tool calls. Some compatible gateways
        # inject tools even when none were supplied. Never execute those tools.
        request_body["tool_choice"] = {"type": "none"}
        if os.environ.get("CLAUDE_TOOL_RECOVERY", "").lower() in {"1", "true", "yes"}:
            request_body["tools"] = []
        if os.environ.get("CLAUDE_STREAM", "").lower() in {"1", "true", "yes"}:
            request_body["stream"] = True
        request = urllib.request.Request(
            url=url,
            data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if request_body.get("stream") else "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "anthropic-version": self.config.api_version,
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                if request_body.get("stream"):
                    return self._read_message_stream(response)
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Claude request failed: {exc}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError(f"Claude connection closed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _read_message_stream(response) -> dict[str, Any]:
        """Accumulate native Messages SSE; usage deltas are cumulative, not sums.

        An interrupted stream is never accepted as a complete generation.
        Unknown events are ignored as required by the upstream API contract.
        """
        message: dict[str, Any] = {"content": [], "usage": {}}
        blocks: dict[int, dict[str, Any]] = {}
        data: list[str] = []
        started = False
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
                continue
            if line or not data:
                continue
            event = json.loads("\n".join(data))
            data = []
            kind = event.get("type")
            if kind == "message_start":
                message = dict(event["message"])
                message["usage"] = dict(message.get("usage") or {})
                started = True
            elif kind == "content_block_start":
                blocks[event["index"]] = dict(event["content_block"])
            elif kind == "content_block_delta":
                delta = event.get("delta", {})
                block = blocks.get(event["index"])
                if block is not None and delta.get("type") == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif block is not None and delta.get("type") == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif block is not None and delta.get("type") == "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.get("signature", "")
                elif block is not None and delta.get("type") == "input_json_delta":
                    block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")
            elif kind == "content_block_stop":
                block = blocks.get(event["index"])
                if block is not None and "_partial_json" in block:
                    block["input"] = json.loads(block.pop("_partial_json"))
            elif kind == "message_delta":
                message.update(event.get("delta") or {})
                message["usage"].update(event.get("usage") or {})
            elif kind == "error":
                error_type = str((event.get("error") or {}).get("type", "unknown"))
                raise RuntimeError(f"Claude streaming upstream_error: {error_type}")
            elif kind == "message_stop":
                if not started:
                    raise RuntimeError("Claude stream missing message_start")
                message["content"] = [blocks[i] for i in sorted(blocks)]
                return message
        raise RuntimeError("Claude connection closed before message_stop")

    @staticmethod
    def _extract_text_from_messages(payload: dict[str, Any]) -> str:
        content = payload.get("content")
        if payload.get("stop_reason") == "tool_use" or (
            isinstance(content, list) and any(isinstance(item, dict) and
                item.get("type") in {"tool_use", "server_tool_use"} for item in content)
        ):
            # Usage has already been recorded by _post_json. Retry within the
            # existing request limit, without accepting partial text or commands.
            raise ClaudeTextResponseError("Claude returned no text output acceptable to text-only adapter: unexpected tool_use")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()
        raise ClaudeTextResponseError("Claude returned no text output acceptable to text-only adapter: empty content")
