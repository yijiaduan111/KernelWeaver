"""OpenAI-compatible provider and shared prompt helpers."""

from __future__ import annotations

import hashlib
import json
import http.client
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..backends import is_cute_backend, is_native_cuda_backend, is_tilelang_backend, normalize_backend
from ..models import AgentContext, AnchorEdit, PlanProposal, SearchNode, TaskSpec
from ..experiment.resource_cost import http_attempt, normalize_usage
from ..semantics import semantic_profile_to_prompt_dict
from ..utils import extract_anchor_names
from .base_provider import AgentProvider


@dataclass(slots=True)
class OpenAICompatibleConfig:
    """Resolved configuration for an OpenAI-compatible backend."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    wire_api: str = "chat_completions"
    max_tokens: int = 8192
    timeout_seconds: int = 300
    plan_temperature: float = 0.7
    code_temperature: float = 0.2
    debug_temperature: float = 0.1
    reasoning_effort: str | None = None
    plan_reasoning_effort: str | None = None
    code_reasoning_effort: str | None = None
    debug_reasoning_effort: str | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    disable_response_storage: bool = False
    user_agent: str = "curl/8.5.0"
    paired_cache_path: str | None = None
    paired_cache_mode: str = "off"
    claude_tool_choice_format: str = "string"


class LLMResponseFormatError(RuntimeError):
    """A successful model response cannot be parsed into the requested format."""


class OpenAICompatibleProvider(AgentProvider):
    """Provider that talks to an OpenAI-compatible API endpoint."""

    name = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def with_overrides(self, **overrides) -> "OpenAICompatibleProvider":
        return OpenAICompatibleProvider(replace(self.config, **overrides))

    @classmethod
    def from_env(
        cls,
        defaults: dict[str, Any] | None = None,
        *,
        env_prefix: str = "OPENAI",
    ) -> "OpenAICompatibleProvider":
        defaults = defaults or {}
        env_prefix = env_prefix.strip().upper()

        def env_name(suffix: str) -> str:
            return f"{env_prefix}_{suffix}"

        api_key = os.environ.get(env_name("API_KEY"), "").strip()
        base_url = str(_env_or_default(env_name("BASE_URL"), defaults.get("base_url", "https://api.openai.com/v1"))).strip().rstrip("/")
        model = str(_env_or_default(env_name("MODEL"), defaults.get("model", "gpt-5.4"))).strip()
        wire_api = str(_env_or_default(env_name("WIRE_API"), defaults.get("wire_api", "chat_completions"))).strip().lower()
        max_tokens = max(1, int(str(_env_or_default(env_name("MAX_TOKENS"), defaults.get("max_tokens", 8192))).strip() or "8192"))
        timeout_seconds = int(str(_env_or_default(env_name("TIMEOUT_SECONDS"), defaults.get("timeout_seconds", 300))).strip() or "300")
        plan_temperature = float(str(_env_or_default(env_name("PLAN_TEMPERATURE"), defaults.get("plan_temperature", 0.7))).strip() or "0.7")
        code_temperature = float(str(_env_or_default(env_name("CODE_TEMPERATURE"), defaults.get("code_temperature", 0.2))).strip() or "0.2")
        debug_temperature = float(str(_env_or_default(env_name("DEBUG_TEMPERATURE"), defaults.get("debug_temperature", 0.1))).strip() or "0.1")
        reasoning_effort = _optional_string(_env_or_default(env_name("REASONING_EFFORT"), defaults.get("reasoning_effort")))
        plan_reasoning_effort = _env_override(
            env_name("PLAN_REASONING_EFFORT"),
            _optional_string(defaults.get("plan_reasoning_effort")) or reasoning_effort,
        )
        code_reasoning_effort = _env_override(
            env_name("CODE_REASONING_EFFORT"),
            _optional_string(defaults.get("code_reasoning_effort")) or reasoning_effort,
        )
        debug_reasoning_effort = _env_override(
            env_name("DEBUG_REASONING_EFFORT"),
            _optional_string(defaults.get("debug_reasoning_effort")) or reasoning_effort,
        )
        max_retries = max(1, int(str(_env_or_default(env_name("MAX_RETRIES"), defaults.get("max_retries", 3))).strip() or "3"))
        retry_backoff_seconds = max(0.0, float(str(_env_or_default(env_name("RETRY_BACKOFF_SECONDS"), defaults.get("retry_backoff_seconds", 1.0))).strip() or "1"))
        disable_response_storage = _env_bool(
            os.environ.get(env_name("DISABLE_RESPONSE_STORAGE")),
            default=bool(defaults.get("disable_response_storage", False)),
        )
        user_agent = str(_env_or_default(env_name("USER_AGENT"), defaults.get("user_agent", "curl/8.5.0"))).strip() or "curl/8.5.0"
        paired_cache_path = _optional_string(_env_or_default(env_name("PAIRED_CACHE_PATH"), defaults.get("paired_cache_path")))
        paired_cache_mode = str(_env_or_default(env_name("PAIRED_CACHE_MODE"), defaults.get("paired_cache_mode", "off"))).strip().lower()
        if paired_cache_mode not in {"off", "record", "replay"}:
            raise ValueError(f"{env_name('PAIRED_CACHE_MODE')} must be off, record, or replay")
        claude_tool_choice_format = str(_env_or_default(
            env_name("CLAUDE_TOOL_CHOICE_FORMAT"), defaults.get("claude_tool_choice_format", "string")
        )).strip().lower()
        if claude_tool_choice_format not in {"string", "object"}:
            raise ValueError(f"{env_name('CLAUDE_TOOL_CHOICE_FORMAT')} must be string or object")
        if not api_key:
            raise ValueError(f"{env_name('API_KEY')} is required for the OpenAI-compatible provider.")
        return cls(
            OpenAICompatibleConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                wire_api=wire_api,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                plan_temperature=plan_temperature,
                code_temperature=code_temperature,
                debug_temperature=debug_temperature,
                reasoning_effort=reasoning_effort,
                plan_reasoning_effort=plan_reasoning_effort,
                code_reasoning_effort=code_reasoning_effort,
                debug_reasoning_effort=debug_reasoning_effort,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                disable_response_storage=disable_response_storage,
                user_agent=user_agent,
                paired_cache_path=paired_cache_path,
                paired_cache_mode=paired_cache_mode,
                claude_tool_choice_format=claude_tool_choice_format,
            )
        )

    def propose_plan(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> PlanProposal:
        anchors = extract_anchor_names(node.code)
        if not anchors:
            raise ValueError("No grounded instruction anchors found in source code.")
        prompt = (
            "You are the planning agent in a STARK-style workflow. Return JSON only.\n"
            "Choose one optimization strategy and decide which code region should be edited.\n"
            "The loader provides generic structural anchors plus a semantic_profile; use the semantic_profile to choose the optimization intent and edit target.\n"
            "Required JSON schema: "
            '{"strategy_name":"...","strategy_summary":"...","expected_gain":"...","risk_notes":"...","anchor_edits":[{"anchor_name":"...","instruction":"...","operation":"replace"}]}\n'
            "Use only anchor names from the provided anchors. operation must be replace or append. "
            "Each instruction must be concrete enough for the coding agent to implement without changing unrelated scaffold."
        ) + _task_prompt_suffix(task, role="plan")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "available_anchors": anchors,
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "leader_nodes": [_snapshot_to_dict(item) for item in context.leaders],
            "related_nodes": [_snapshot_to_dict(item) for item in context.related],
            "task_experience_memory": _long_term_to_payload(context),
            "current_code": node.code,
        }
        if context.workflow_guidance:
            prompt += (
                "\nFollow the supplied KDA-style evidence workflow. Select exactly one primary "
                "performance hypothesis for this candidate. In strategy_summary, state the "
                "evidence, the hypothesized bottleneck, and why this edit should improve it. "
                "In expected_gain, state the measurable outcome used to accept or reject it. "
                "Do not import architecture-specific assumptions that are not observed in the "
                "current run. Correctness validation and measured runtime are the only promotion criteria."
            )
            user["workflow_guidance"] = context.workflow_guidance
        operator_memory = list(context.operator_memory or context.domain_wiki_evidence)
        if operator_memory:
            prompt += (
                "\nThe supplied long_term_operator_memory is a curated, architecture-scoped prior. "
                "It is not permission to change the task contract or build path. Apply at most "
                "one matching hypothesis, obey every avoid/global constraint, and keep the existing "
                "extension loader and grounded scaffold unchanged. Do not introduce a second loader, "
                "dependency, DSL, or unsupported architecture feature. Correctness and measured "
                "runtime against the root are the only promotion criteria."
            )
            user["long_term_operator_memory"] = operator_memory
            if any(item.get("knowledge_scope") == "public_general" for item in operator_memory):
                prompt += (
                    "\nPublic Wiki review verifies the cited documentation only, not a kernel's "
                    "correctness or measured speed. Check each applicability condition against "
                    "this task. Treat inferred optimization advice and API co-occurrence as "
                    "hypotheses, and report missing specialized coverage honestly."
                )
        if context.performance_feedback:
            prompt += (
                "\nHardware feedback is one sampled launch of an earlier validated candidate, "
                "not a claim about all shapes or this current code. Use its observed pressure "
                "to propose a concrete algorithm or data-movement improvement only when relevant; "
                "do not default to changing launch threads. For a single-pass elementwise kernel "
                "that already reads each input once and writes each output once, shared memory "
                "does not lower compulsory global bytes; vector loads do not lower bytes by "
                "themselves. Check the actual access pattern before suggesting either. "
                "Preserve correctness constraints."
            )
            user["hardware_feedback"] = context.performance_feedback
        data = self._chat_json_object(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.plan_temperature,
            reasoning_effort=self.config.plan_reasoning_effort,
            required_keys=("strategy_name", "strategy_summary", "expected_gain", "anchor_edits"),
        )
        edits = [
            AnchorEdit(
                anchor_name=item["anchor_name"],
                instruction=item["instruction"],
                operation=_normalize_operation(item.get("operation", "replace")),
            )
            for item in data.get("anchor_edits") or []
        ]
        return PlanProposal(
            strategy_name=data["strategy_name"],
            strategy_summary=data["strategy_summary"],
            anchor_edits=edits,
            expected_gain=data["expected_gain"],
            risk_notes=data.get("risk_notes", ""),
        )

    def generate_search_candidate(
        self,
        task: TaskSpec,
        node: SearchNode,
        context: AgentContext,
    ) -> tuple[PlanProposal, str]:
        prompt = (
            "You are the single Search Agent in a STARK ablation. Return JSON only.\n"
            "Do not split planning, coding, or debugging into separate roles. Generate the next full candidate directly.\n"
            'Required JSON schema: {"strategy_name":"...","strategy_summary":"...","expected_gain":"...","risk_notes":"...","code":"..."}'
        ) + _task_prompt_suffix(task, role="search")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "current_code": node.code,
            "logs": node.logs,
            "failure_stage": node.latest_failure_stage,
        }
        data = self._chat_json_object(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.code_temperature,
            reasoning_effort=self.config.code_reasoning_effort,
            required_keys=("code",),
        )
        proposal = PlanProposal(
            strategy_name=data.get("strategy_name", "search-agent"),
            strategy_summary=data.get("strategy_summary", "Single-agent candidate generation."),
            anchor_edits=[],
            expected_gain=data.get("expected_gain", "unknown"),
            risk_notes=data.get("risk_notes", ""),
        )
        return proposal, _strip_code_fences(str(data["code"]))

    def generate_code(
        self,
        task: TaskSpec,
        node: SearchNode,
        proposal: PlanProposal,
        context: AgentContext,
    ) -> str:
        prompt = (
            "You are the coding agent in a STARK-style workflow.\n"
            "Return only the full updated Python source code. No markdown fences.\n"
            "Preserve function signatures and task I/O. Apply the plan through the provided grounded anchor edits.\n"
            "Use semantic_profile only as an implementation hint; do not change task semantics or evaluator I/O.\n"
            "Do not delete, rename, or move any existing # <<<IMPROVE:...>>> or # <<<END_IMPROVE>>> anchor markers."
        ) + _task_prompt_suffix(task, role="code")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "available_anchors": extract_anchor_names(node.code),
            "plan": {
                "strategy_name": proposal.strategy_name,
                "strategy_summary": proposal.strategy_summary,
                "expected_gain": proposal.expected_gain,
                "risk_notes": proposal.risk_notes,
                "anchor_edits": [
                    {
                        "anchor_name": edit.anchor_name,
                        "instruction": edit.instruction,
                        "operation": edit.operation,
                    }
                    for edit in proposal.anchor_edits
                ],
            },
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "related_nodes": [_snapshot_to_dict(item) for item in context.related],
            "task_experience_memory": _long_term_to_payload(context),
            "current_code": node.code,
        }
        content = self._chat(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.code_temperature,
            reasoning_effort=self.config.code_reasoning_effort,
        )
        return _strip_code_fences(content)

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        debug_focus = _debug_focus_hint(node.latest_failure_stage, task.backend)
        prompt = (
            "You are the debug agent in a STARK-style workflow.\n"
            "Return only the full corrected Python source code. No markdown fences.\n"
            "Apply the smallest local fix needed to recover compilation, runtime, or correctness.\n"
            "Do not delete, rename, or move any existing # <<<IMPROVE:...>>> or # <<<END_IMPROVE>>> anchor markers.\n"
            f"{debug_focus}"
        ) + _task_prompt_suffix(task, role="debug")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "related_nodes": [_snapshot_to_dict(item) for item in context.related],
            "task_experience_memory": _long_term_to_payload(context),
            "failure_node": _snapshot_to_dict(context.failure) if context.failure else None,
            "plan_strategy_name": node.plan_strategy_name,
            "plan_summary": node.plan_summary,
            "failure_stage": node.latest_failure_stage,
            "logs": node.logs,
            "failing_code": node.code,
        }
        content = self._chat(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.debug_temperature,
            reasoning_effort=self.config.debug_reasoning_effort,
        )
        return _strip_code_fences(content)

    def _chat(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        cache_key = self._paired_cache_key(system_prompt, user_payload, temperature)
        cached = self._paired_cache_read(cache_key)
        if cached is not None:
            return cached
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            try:
                if self.config.wire_api == "responses":
                    payload = self._responses_request(system_prompt, user_payload, temperature, reasoning_effort)
                    content = self._extract_text_from_responses(payload)
                else:
                    payload = self._chat_completions_request(system_prompt, user_payload, temperature)
                    content = self._extract_text_from_chat_completions(payload)
                self._paired_cache_write(cache_key, content)
                return content
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
        raise RuntimeError("LLM request failed before receiving a response")

    def _paired_cache_key(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
    ) -> str | None:
        if not self.config.paired_cache_path or self.config.paired_cache_mode == "off":
            return None
        # Cross-candidate context is the treatment. Calls are paired only while
        # no related/leader memory is visible to the agent.
        if user_payload.get("related_nodes") or user_payload.get("leader_nodes"):
            return None
        payload = {
            "model": self.config.model,
            "wire_api": self.config.wire_api,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "user_payload": _without_volatile_measurements(user_payload),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _paired_cache_read(self, cache_key: str | None) -> str | None:
        if cache_key is None or self.config.paired_cache_mode != "replay":
            return None
        entries = self._load_paired_cache().get(cache_key) or []
        counters = getattr(self, "_paired_cache_counters", None)
        if counters is None:
            counters = {}
            self._paired_cache_counters = counters
        index = int(counters.get(cache_key, 0))
        if index >= len(entries):
            return None
        counters[cache_key] = index + 1
        return str(entries[index])

    def _paired_cache_write(self, cache_key: str | None, content: str) -> None:
        if cache_key is None or self.config.paired_cache_mode != "record":
            return
        path = Path(str(self.config.paired_cache_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_paired_cache()
        payload.setdefault(cache_key, []).append(content)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        self._paired_cache_payload = payload

    def _load_paired_cache(self) -> dict[str, list[str]]:
        cached = getattr(self, "_paired_cache_payload", None)
        if isinstance(cached, dict):
            return cached
        path = Path(str(self.config.paired_cache_path))
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = {
                str(key): [str(item) for item in value]
                for key, value in raw.items()
                if isinstance(value, list)
            }
        else:
            payload = {}
        self._paired_cache_payload = payload
        return payload

    def _chat_json_object(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None = None,
        required_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Retry model calls whose text is not a usable JSON object.

        Some OpenAI-compatible gateways return HTTP 200 even when an upstream
        Claude response is truncated or contains prose instead of the requested
        JSON. Treat those responses like retryable LLM failures so a single bad
        completion does not discard an entire benchmark task.
        """
        last_error: RuntimeError | None = None
        retry_system_prompt = system_prompt
        retry_user_payload = user_payload
        for attempt in range(max(1, self.config.max_retries)):
            try:
                content = self._chat(
                    retry_system_prompt,
                    retry_user_payload,
                    temperature if attempt == 0 else 0.0,
                    reasoning_effort,
                )
                payload = _parse_json_object(content)
                missing = [key for key in required_keys if key not in payload]
                if missing:
                    raise LLMResponseFormatError(f"LLM JSON response is missing required keys: {', '.join(missing)}")
                return payload
            except (LLMResponseFormatError, json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc))
                invalid_output = locals().get("content", "")
                retry_system_prompt = (
                    "CORRECTION REQUIRED: the previous answer was not valid for the requested JSON schema. "
                    "Reply with exactly one valid JSON object. The first non-whitespace character must be '{' "
                    "and the last must be '}'. Do not add an introduction, explanation, or markdown fence.\n\n"
                    + system_prompt
                )
                retry_user_payload = dict(user_payload)
                if isinstance(invalid_output, str) and invalid_output.strip():
                    retry_user_payload["previous_invalid_response"] = invalid_output[:2000]
            if attempt + 1 < self.config.max_retries and self.config.retry_backoff_seconds > 0:
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM failed to return a JSON object")

    def _chat_completions_request(self, system_prompt: str, user_payload: dict[str, Any], temperature: float) -> dict[str, Any]:
        request_body = {
            "model": self.config.model,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": _compact_payload_text(user_payload)}],
        }
        if self.config.model.lower().startswith("claude-"):
            # Micu's Claude route accepts the OpenAI-compatible path but maps
            # system instructions to Anthropic's top-level `system` field.
            request_body["system"] = system_prompt
            # Some routed Claude accounts inject a tool-enabled upstream
            # environment even though this client sends no tool definitions.
            # Explicitly disable tool use so JSON/code agents return text
            # instead of an empty message containing a synthetic Bash call.
            request_body["tool_choice"] = (
                {"type": "none"} if self.config.claude_tool_choice_format == "object" else "none"
            )
        else:
            request_body["messages"].insert(0, {"role": "system", "content": system_prompt})
        return self._post_json(self._build_endpoint("chat_completions"), request_body)

    def _responses_request(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        _ = temperature
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": _compact_payload_text(user_payload)}],
                },
            ],
        }
        optional_fields: list[tuple[str, Any]] = []
        if reasoning_effort:
            optional_fields.append(("reasoning", {"effort": reasoning_effort}))
        if self.config.disable_response_storage:
            optional_fields.append(("store", False))
        for key, value in optional_fields:
            request_body[key] = value
        last_error: RuntimeError | None = None
        while True:
            try:
                return self._post_json(self._build_endpoint("responses"), request_body)
            except RuntimeError as exc:
                last_error = exc
                if "Unsupported parameter" not in str(exc) or not optional_fields:
                    raise
                key, _value = optional_fields.pop()
                request_body.pop(key, None)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed before receiving a response")

    def _post_json(self, url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        with http_attempt(str(request_body.get("model", self.config.model))) as event:
            payload = self._post_json_unmetered(url, request_body)
            event.update(normalize_usage(payload))
            return payload

    def _post_json_unmetered(self, url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8", errors="replace")
                if not raw_body.strip():
                    raise RuntimeError("LLM request returned an empty HTTP response body")
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    preview = raw_body[:200].replace("\n", "\\n")
                    raise RuntimeError(f"LLM request returned invalid JSON: {preview}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError("LLM request returned JSON that is not an object")
                return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def _build_endpoint(self, wire_api: str) -> str:
        base = self.config.base_url.rstrip("/")
        if not re.search(r"/v\d+$", base):
            base = f"{base}/v1"
        if wire_api == "responses":
            return f"{base}/responses"
        return f"{base}/chat/completions"

    @staticmethod
    def _extract_text_from_chat_completions(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response does not contain choices: {payload}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"LLM response content is empty: {payload}")
        return content.strip()

    @staticmethod
    def _extract_text_from_responses(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()
        collected: list[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("content"), list):
                for content_item in item["content"]:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        collected.append(text.strip())
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())
        if collected:
            return "\n".join(collected)
        raise RuntimeError(f"Responses API returned no text output: {payload}")


def _snapshot_to_dict(snapshot) -> dict[str, Any]:
    return {
        "node_id": snapshot.node_id,
        "parent_id": snapshot.parent_id,
        "depth": snapshot.depth,
        "score": snapshot.score,
        "status": snapshot.status,
        "plan_strategy_name": snapshot.plan_strategy_name,
        "failure_type": snapshot.failure_type,
        "child_count": snapshot.child_count,
        "origin": snapshot.origin,
        "selected_count": snapshot.selected_count,
        "runtime": snapshot.runtime,
        "latest_failure_stage": snapshot.latest_failure_stage,
        "reference_runtime": getattr(snapshot, "reference_runtime", None),
        "speedup": getattr(snapshot, "speedup", None),
        "delta_vs_root": getattr(snapshot, "delta_vs_root", None),
        "delta_vs_parent": getattr(snapshot, "delta_vs_parent", None),
        "failure_log_excerpt": getattr(snapshot, "failure_log_excerpt", None),
        "code_hash": getattr(snapshot, "code_hash", None),
        "plan_summary": getattr(snapshot, "plan_summary", None),
        "anchor_edits": getattr(snapshot, "anchor_edits", []),
        "changed_anchor_excerpts": getattr(snapshot, "changed_anchor_excerpts", {}),
        "memory_lesson": getattr(snapshot, "memory_lesson", None),
    }


def _long_term_to_payload(context: AgentContext) -> list[dict[str, Any]]:
    return [
        {
            "task_name": item.get("task_name"),
            "op_type": item.get("op_type"),
            "outcome": item.get("outcome"),
            "failure_stage": item.get("failure_stage"),
            "failure_type": item.get("failure_type"),
            "speedup": item.get("speedup"),
            "lesson": item.get("lesson"),
        }
        for item in getattr(context, "long_term", [])
    ]


def _task_metadata(task: TaskSpec) -> dict[str, Any]:
    return {
        "benchmark_family": task.benchmark_family,
        "entry_kind": task.entry_kind,
        "level": task.level,
        "problem_id": task.problem_id,
        "backend": task.backend,
        "source_origin": task.source_origin,
        "semantic_profile": semantic_profile_to_prompt_dict(task.semantic_profile),
        "grounded_regions": [
            {
                "anchor_name": region.anchor_name,
                "region_role": region.region_role,
                "start_line": region.start_line,
                "end_line": region.end_line,
                "source_excerpt": region.source_excerpt,
                "source_hash": region.source_hash,
            }
            for region in task.grounded_regions
        ],
    }

def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise LLMResponseFormatError("LLM response text is empty; expected a JSON object")
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise LLMResponseFormatError("LLM response JSON is not an object")
        return payload
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", cleaned):
            try:
                payload, _end = decoder.raw_decode(cleaned[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        preview = cleaned[:200].replace("\n", "\\n")
        raise LLMResponseFormatError(f"LLM response does not contain a valid JSON object: {preview}")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _task_prompt_suffix(task: TaskSpec, role: str) -> str:
    if task.benchmark_family == "kernelbench":
        backend_hint = _kernelbench_backend_hint(task, role)
        base = "\nThis task comes from the real KernelBench benchmark and is evaluated on CUDA. "
        anchor_hint = _kernelbench_anchor_hint(task)
        profile_hint = _kernelbench_profile_hint(task, role)
        if role == "search":
            return (
                base
                + "Return exactly the JSON object required by the Search Agent schema, without markdown fences or prose. "
                + "Put the full executable Python candidate in the JSON code field, preserve the public evaluator I/O, keep ModelNew as the entry class when present, "
                + "and keep any existing scaffold structure stable unless the current code is already broken."
                + backend_hint
                + profile_hint
            )
        if role == "plan":
            return (
                base
                + "Return exactly the JSON object required by the planning schema, without markdown fences, prose, or Python source outside JSON. "
                + f"Select the most relevant edit target from the provided structural anchors ({anchor_hint}). "
                + "Do not assume any handwritten adapter has pre-selected the target; infer the best region from the raw KernelBench ModelNew code and current feedback. "
                + "Preserve the generated scaffold and keep ModelNew as the entry class."
                + backend_hint
                + profile_hint
            )
        return (
            base
            + "Return executable Python code only and do not include markdown fences or prose. "
            + "The code must define ModelNew, keep the scaffold outside the anchors unchanged, preserve the __init__/forward signatures, and restrict edits to the provided anchors."
            + backend_hint
            + profile_hint
        )
    if "triton" not in task.tags:
        return ""
    if role == "search":
        return (
            "\nThis task targets a real Triton GPU kernel on CUDA tensors. "
            "Return one full executable Python candidate using torch and triton, preserve the task entrypoint signature, "
            "and do not include markdown fences or explanations."
        )
    if role == "plan":
        return (
            "\nThis task targets a real Triton GPU kernel on CUDA tensors. "
            "Prefer strategies that compile as Python + Triton code and stay within the grounded anchors."
        )
    return (
        "\nThis task targets a real Triton GPU kernel on CUDA tensors. "
        "The output must be executable Python source using torch and triton, preserve the task entrypoint signature, "
        "preserve all existing anchor markers, and must not include markdown fences or explanations."
    )


def _kernelbench_anchor_hint(task: TaskSpec) -> str:
    anchors = extract_anchor_names(task.source_code)
    if is_native_cuda_backend(task.backend):
        return "helpers/cuda_cpp/cuda_cu/init_body/forward_stmt_* anchors"
    if is_tilelang_backend(task.backend):
        if any(anchor.startswith("forward_stmt_") for anchor in anchors):
            return "helpers/tilelang_kernel/init_body/forward_stmt_* anchors"
        return "helpers/tilelang_kernel/init_body/forward_stmt_* anchors"
    if is_cute_backend(task.backend):
        if any(anchor.startswith("forward_stmt_") for anchor in anchors):
            return "helpers/cute_kernel/init_body/forward_stmt_* anchors"
        return "helpers/cute_kernel/init_body/forward_stmt_* anchors"
    if any(anchor.startswith("forward_stmt_") for anchor in anchors):
        return "helpers/init_body/forward_stmt_* anchors"
    return "helpers/init_body/forward_stmt_* anchors"


def _kernelbench_backend_hint(task: TaskSpec, role: str) -> str:
    backend = normalize_backend(task.backend)
    if is_native_cuda_backend(backend):
        return ""
    if is_tilelang_backend(backend):
        if role == "plan":
            return (
                " Treat this as a TileLang task: keep ModelNew and evaluator I/O unchanged, "
                "put TileLang-specific helpers inside helpers/tilelang_kernel, and plan localized edits only."
            )
        return (
            " Treat this as a TileLang task: return executable Python using torch and tilelang, "
            "keep ModelNew and evaluator I/O unchanged, and place TileLang imports/helpers inside the grounded scaffold."
        )
    if is_cute_backend(backend):
        if role == "plan":
            return (
                " Treat this as a CuTe DSL task: keep ModelNew and evaluator I/O unchanged, "
                "put CuTe-specific helpers inside helpers/cute_kernel, and plan localized edits only."
            )
        return (
            " Treat this as a CuTe DSL task: return executable Python using torch and cutlass.cute, "
            "keep ModelNew and evaluator I/O unchanged, and place CuTe imports/helpers inside the grounded scaffold."
        )
    if backend == "triton":
        if role == "plan":
            return (
                " Treat this as a Triton task: keep ModelNew and evaluator I/O unchanged, "
                "and prefer Triton-friendly grounded edits over plain eager PyTorch rewrites."
            )
        return (
            " Treat this as a Triton task: return executable Python using torch and triton, "
            "keep ModelNew and evaluator I/O unchanged, and preserve the grounded scaffold."
        )
    return ""


def _kernelbench_profile_hint(task: TaskSpec, role: str) -> str:
    tags = set(task.tags)
    if is_native_cuda_backend(task.backend):
        index_safety = (
            " Use int64_t (or another proven 64-bit type) for numel, flattened indices, offsets, strides, and their intermediate arithmetic; "
            "KernelBench tensors can exceed INT32_MAX elements. Bounds-check every global-memory access and surface launch errors promptly."
        )
        if role == "plan":
            return (
                " Treat this as a native CUDA extension task: keep the scaffold intact, preserve the ModelNew interface, "
                "and propose grounded edits that stay inside helpers/cuda_cpp/cuda_cu/init_body/forward_stmt_*."
                + index_safety
            )
        return (
            " Treat this as a native CUDA extension task: preserve the pybind binding surface, keep CUDA kernel launch assumptions explicit, "
            "and restrict edits to the grounded regions only."
            + index_safety
        )
    if "level3" in tags:
        if "attention" in tags:
            return (
                " Treat this as a Level 3 attention block: preserve reshape and permute order, keep attention query/key/value usage exact, "
                "and keep residual-plus-normalization math inside the grounded anchors."
            )
        if "cnn_block" in tags or "residual" in tags:
            return (
                " Treat this as a Level 3 CNN block: preserve residual connections, downsample behavior, convolution shapes, "
                "and activation ordering while editing only the grounded anchors."
            )
        if "mlp" in tags:
            return (
                " Treat this as a Level 3 MLP block: preserve the Sequential module structure, hidden sizes, and final output shape, "
                "and keep any explicit step rewrites inside the grounded anchors."
            )
    if "conv" in tags:
        return (
            " Treat this as a conv-fusion task: preserve the Conv module state, keep tensor layout valid for CUDA, "
            "and only rewrite the post-conv math inside the grounded anchors."
        )
    if "gemm" in tags or "fusion" in tags:
        return (
            " Treat this as a GEMM-style fusion task: preserve module parameters, keep linear/bias dimensions exact, "
            "and express fused math explicitly inside the anchors."
        )
    if "norm" in tags or "reduction" in tags or "elementwise" in tags or "parameterized" in tags:
        focus = "preserve normalization statistics and affine state" if "norm" in tags else "preserve reduction dimension and output shape"
        if "elementwise" in tags or "parameterized" in tags:
            focus = "preserve exact activation math, broadcasting, and parameter usage"
        return f" Treat this as a unary/norm/reduction task: {focus}, and keep the generated scaffold outside the anchors unchanged."
    if role == "debug":
        return " Keep fixes local to the failing anchor and preserve the official task I/O exactly."
    return ""


def _normalize_operation(value: str) -> str:
    normalized = (value or "replace").strip().lower()
    return normalized or "replace"


def _debug_focus_hint(failure_stage: str | None, backend: str | None = None) -> str:
    if is_native_cuda_backend(backend):
        if failure_stage == "compile":
            return "Prioritize C++ binding signatures, CUDA kernel signatures, includes, extension build settings, and pybind exports."
        if failure_stage == "runtime":
            return "Prioritize CUDA device placement, grid/block sizing, pointer math, tensor shapes and strides, and contiguous assumptions."
        if failure_stage == "correctness":
            return "Prioritize formulas, reduction dimensions, masks, write-back indices, and broadcasting semantics."
        return "Prioritize the smallest local fix while preserving the native CUDA extension scaffold."
    if is_tilelang_backend(backend):
        if failure_stage == "compile":
            return "Prioritize TileLang imports, kernel definition syntax, buffer shape annotations, and launcher wiring."
        if failure_stage == "runtime":
            return "Prioritize launch shapes, buffer layouts, memory scopes, tensor contiguity, and CUDA device placement."
        if failure_stage == "correctness":
            return "Prioritize TileLang indexing, masks, reduction axes, and write-back layout."
        return "Prioritize the smallest local fix while preserving the TileLang scaffold."
    if is_cute_backend(backend):
        if failure_stage == "compile":
            return "Prioritize CuTe imports, decorators, tensor layout construction, and kernel launcher wiring."
        if failure_stage == "runtime":
            return "Prioritize CuTe tensor layouts, launch arguments, memory movement, and CUDA device placement."
        if failure_stage == "correctness":
            return "Prioritize CuTe index mapping, tile partitioning, reduction axes, and write-back layout."
        return "Prioritize the smallest local fix while preserving the CuTe scaffold."
    if failure_stage == "compile":
        return "Prioritize syntax fixes, imports, Triton decorators, and function signatures."
    if failure_stage == "runtime":
        return "Prioritize CUDA device placement, Triton launch configuration, pointer math, tensor shapes, and masks."
    if failure_stage == "correctness":
        return "Prioritize math formulas, indexing, masks, reduction logic, and output layout."
    return "Prioritize the smallest localized fix consistent with the existing plan."


def _compact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            compact_item = _compact_payload(item)
            if compact_item in (None, "", [], {}):
                continue
            compacted[key] = compact_item
        return compacted
    if isinstance(value, list):
        compacted_items = [_compact_payload(item) for item in value]
        return [item for item in compacted_items if item not in (None, "", [], {})]
    return value


def _compact_payload_text(payload: dict[str, Any]) -> str:
    return json.dumps(_compact_payload(payload), ensure_ascii=False, separators=(",", ":"))


def _without_volatile_measurements(value: Any) -> Any:
    """Remove evaluator noise from keys used to pair no-memory LLM calls."""
    volatile = {
        "score",
        "runtime",
        "reference_runtime",
        "speedup",
        "delta_vs_root",
        "delta_vs_parent",
        "selected_count",
    }
    if isinstance(value, dict):
        return {
            key: _without_volatile_measurements(item)
            for key, item in value.items()
            if key not in volatile
        }
    if isinstance(value, list):
        return [_without_volatile_measurements(item) for item in value]
    return value


def _env_override(key: str, default: str | None) -> str | None:
    if key not in os.environ:
        return default
    value = os.environ.get(key, "").strip()
    return value or None


def _env_or_default(key: str, default: Any) -> Any:
    if key not in os.environ:
        return default
    return os.environ.get(key, default)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_retryable_llm_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    retryable_tokens = (
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "upstream_error",
        "temporarily unavailable",
        "timed out",
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "remote end closed connection",
        "connection reset by peer",
        "connection aborted",
        "connection closed",
        "response content is empty",
        "returned no text output",
        "empty http response body",
        "ssl",
    )
    return any(token in message for token in retryable_tokens)


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_torch_dtype(torch_module, dtype_name: str):
    normalized = str(dtype_name or "bfloat16").strip().lower()
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    return mapping.get(normalized, torch_module.bfloat16)


def _compose_local_chat_prompt(tokenizer, messages: list[dict[str, str]], use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        parts.append(f"{role}:\n{content}")
    parts.append("Assistant:\n")
    return "\n\n".join(parts)


__all__ = [
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "_compact_payload_text",
]
