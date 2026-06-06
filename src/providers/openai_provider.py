"""OpenAI-compatible provider and shared prompt helpers."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from ..backends import is_cute_backend, is_native_cuda_backend, is_tilelang_backend, normalize_backend
from ..core.patch_payload import canonicalize_region_patches, parse_loose_json_dict
from ..deliberation import strategy_portfolio_to_prompt_dict
from ..models import AgentContext, AnchorEdit, PlanProposal, SearchNode, TaskSpec
from ..semantics import semantic_profile_to_prompt_dict
from ..utils import extract_anchor_names
from .base_provider import AgentProvider


@dataclass
class OpenAICompatibleConfig:
    """Resolved configuration for an OpenAI-compatible backend."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    wire_api: str = "chat_completions"
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
    responses_fallback_to_chat_completions: bool = True
    user_agent: str = "curl/8.5.0"


class OpenAICompatibleProvider(AgentProvider):
    """Provider that talks to an OpenAI-compatible API endpoint."""

    name = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = self._normalize_config(config)

    def with_overrides(self, **overrides) -> "OpenAICompatibleProvider":
        return OpenAICompatibleProvider(replace(self.config, **overrides))

    @staticmethod
    def _normalize_config(config: OpenAICompatibleConfig) -> OpenAICompatibleConfig:
        if config.wire_api == "chat_completions" and not config.responses_fallback_to_chat_completions:
            return config
        return replace(
            config,
            wire_api="chat_completions",
            responses_fallback_to_chat_completions=False,
        )

    def generate_text(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.2,
        purpose: str = "generic",
    ) -> str:
        del purpose
        return self._chat(system_prompt, user_payload, temperature, reasoning_effort=None)

    @classmethod
    def from_env(cls, defaults: dict[str, Any] | None = None) -> "OpenAICompatibleProvider":
        defaults = defaults or {}
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = str(_env_or_default("OPENAI_BASE_URL", defaults.get("base_url", "https://api.openai.com/v1"))).strip().rstrip("/")
        model = str(_env_or_default("OPENAI_MODEL", defaults.get("model", "gpt-5.4"))).strip()
        wire_api = str(_env_or_default("OPENAI_WIRE_API", defaults.get("wire_api", "chat_completions"))).strip().lower()
        timeout_seconds = int(str(_env_or_default("OPENAI_TIMEOUT_SECONDS", defaults.get("timeout_seconds", 300))).strip() or "300")
        plan_temperature = float(str(_env_or_default("OPENAI_PLAN_TEMPERATURE", defaults.get("plan_temperature", 0.7))).strip() or "0.7")
        code_temperature = float(str(_env_or_default("OPENAI_CODE_TEMPERATURE", defaults.get("code_temperature", 0.2))).strip() or "0.2")
        debug_temperature = float(str(_env_or_default("OPENAI_DEBUG_TEMPERATURE", defaults.get("debug_temperature", 0.1))).strip() or "0.1")
        reasoning_effort = _optional_string(_env_or_default("OPENAI_REASONING_EFFORT", defaults.get("reasoning_effort")))
        plan_reasoning_effort = _env_override(
            "OPENAI_PLAN_REASONING_EFFORT",
            _optional_string(defaults.get("plan_reasoning_effort")) or reasoning_effort,
        )
        code_reasoning_effort = _env_override(
            "OPENAI_CODE_REASONING_EFFORT",
            _optional_string(defaults.get("code_reasoning_effort")) or reasoning_effort,
        )
        debug_reasoning_effort = _env_override(
            "OPENAI_DEBUG_REASONING_EFFORT",
            _optional_string(defaults.get("debug_reasoning_effort")) or reasoning_effort,
        )
        max_retries = max(1, int(str(_env_or_default("OPENAI_MAX_RETRIES", defaults.get("max_retries", 3))).strip() or "3"))
        retry_backoff_seconds = max(0.0, float(str(_env_or_default("OPENAI_RETRY_BACKOFF_SECONDS", defaults.get("retry_backoff_seconds", 1.0))).strip() or "1"))
        disable_response_storage = _env_bool(
            os.environ.get("OPENAI_DISABLE_RESPONSE_STORAGE"),
            default=bool(defaults.get("disable_response_storage", False)),
        )
        responses_fallback_to_chat_completions = _env_bool(
            os.environ.get("OPENAI_RESPONSES_FALLBACK_TO_CHAT_COMPLETIONS"),
            default=bool(defaults.get("responses_fallback_to_chat_completions", True)),
        )
        user_agent = str(_env_or_default("OPENAI_USER_AGENT", defaults.get("user_agent", "curl/8.5.0"))).strip() or "curl/8.5.0"
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai-compatible provider.")
        return cls(
            OpenAICompatibleConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                wire_api=wire_api,
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
                responses_fallback_to_chat_completions=responses_fallback_to_chat_completions,
                user_agent=user_agent,
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
            "Each instruction must be concrete enough for the coding agent to implement without changing unrelated scaffold. "
            "If task_metadata.strategy_portfolio is present, set strategy_name to one selected strategy_id from it, "
            "Use strategy_history to guide selection: prefer strategies not yet attempted; if a strategy achieved speedup > 1.0, consider refinement variants; avoid strategies with only compile failures unless you have a concrete fix."
        ) + _task_prompt_suffix(task, role="plan")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "available_anchors": anchors,
            "strategy_history": context.strategy_history,
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "leader_nodes": [_snapshot_to_dict(item) for item in context.leaders],
            "related_nodes": [_snapshot_to_dict(item) for item in context.related],
            "current_code": node.code,
        }
        content = self._chat(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.plan_temperature,
            reasoning_effort=self.config.plan_reasoning_effort,
        )
        data = _parse_json_object(content)
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
        content = self._chat(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.code_temperature,
            reasoning_effort=self.config.code_reasoning_effort,
        )
        data = _parse_json_object(content)
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
            "You are the coding agent in a STARK-style workflow. Return JSON only.\n"
            "Do not return a full Python file and do not include markdown fences.\n"
            "Required JSON schema: "
            '{"region_patches":[{"region":"...","operation":"replace","body":"..."}]}\n'
            "Edit only the requested editable regions. Each body contains code inside that region only.\n"
            "Never include # <<<IMPROVE:...>>> or # <<<END_IMPROVE>>> marker comments in a body.\n"
            "You may write complete CUDA kernels, helper functions, launchers, pybind bindings, and fallback logic inside their regions.\n"
            "Preserve task semantics, ModelNew/forward signatures, evaluator I/O, and protected scaffold.\n"
            "Use semantic_profile and strategy_portfolio only as implementation hints."
        ) + _task_prompt_suffix(task, role="code")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "available_anchors": extract_anchor_names(node.code),
            "available_regions": extract_anchor_names(node.code),
            "requested_regions": [edit.anchor_name for edit in proposal.anchor_edits],
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
            "current_code": node.code,
        }
        content = self._chat(
            system_prompt=prompt,
            user_payload=user,
            temperature=self.config.code_temperature,
            reasoning_effort=self.config.code_reasoning_effort,
        )
        return _normalize_patch_response(
            response_text=content,
            allowed_regions={edit.anchor_name: edit.operation for edit in proposal.anchor_edits},
        )

    def debug_code(self, task: TaskSpec, node: SearchNode, context: AgentContext) -> str:
        debug_focus = _debug_focus_hint(node.latest_failure_stage, task.backend)
        prompt = (
            "You are the debug agent in a STARK-style workflow. Return JSON only.\n"
            "Do not return a full Python file and do not include markdown fences.\n"
            "Required JSON schema: "
            '{"region_patches":[{"region":"...","operation":"replace","body":"..."}]}\n'
            "Apply the smallest local fix needed inside editable regions only.\n"
            "Never include # <<<IMPROVE:...>>> or # <<<END_IMPROVE>>> marker comments in a body.\n"
            "Preserve task semantics, ModelNew/forward signatures, evaluator I/O, and protected scaffold.\n"
            f"{debug_focus}"
        ) + _task_prompt_suffix(task, role="debug")
        user = {
            "task_name": task.name,
            "task_description": task.description,
            "task_metadata": _task_metadata(task),
            "current_node": _snapshot_to_dict(context.current),
            "root_node": _snapshot_to_dict(context.root),
            "related_nodes": [_snapshot_to_dict(item) for item in context.related],
            "failure_node": _snapshot_to_dict(context.failure) if context.failure else None,
            "available_regions": extract_anchor_names(node.code),
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
        return _normalize_patch_response(
            response_text=content,
            allowed_regions={name: "replace" for name in extract_anchor_names(node.code)},
        )

    def _chat(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            try:
                if self.config.wire_api == "responses":
                    payload = self._responses_request(system_prompt, user_payload, temperature, reasoning_effort)
                    try:
                        return self._extract_text_from_responses(payload)
                    except RuntimeError as exc:
                        if self._should_fallback_from_responses(payload, exc):
                            payload = self._chat_completions_request(system_prompt, user_payload, temperature)
                            return self._extract_text_from_chat_completions(payload)
                        raise
                payload = self._chat_completions_request(system_prompt, user_payload, temperature)
                return self._extract_text_from_chat_completions(payload)
            except TimeoutError as exc:
                last_error = exc
            except RuntimeError as exc:
                last_error = exc
                if not _is_retryable_llm_error(exc):
                    raise
            if attempt + 1 < attempts:
                delay_seconds = _retry_delay_seconds(exc=last_error, attempt_index=attempt, default_backoff=self.config.retry_backoff_seconds)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed before receiving a response")

    def _chat_completions_request(self, system_prompt: str, user_payload: dict[str, Any], temperature: float) -> dict[str, Any]:
        request_body = {
            "model": self.config.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _compact_payload_text(user_payload)},
            ],
        }
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
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def _build_endpoint(self, wire_api: str) -> str:
        base = self.config.base_url.rstrip("/")
        if not re.search(r"/v\d+$", base):
            base = f"{base}/v1"
        if wire_api == "responses":
            return f"{base}/responses"
        return f"{base}/chat/completions"

    def _should_fallback_from_responses(self, payload: dict[str, Any], exc: RuntimeError) -> bool:
        if not self.config.responses_fallback_to_chat_completions:
            return False
        if self.config.wire_api != "responses":
            return False
        message = str(exc)
        if "Responses API returned no text output" not in message:
            return False
        return str(payload.get("status", "")).lower() == "completed"

    @staticmethod
    def _extract_text_from_chat_completions(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response does not contain choices: {payload}")
        message = choices[0].get("message") or {}
        content = OpenAICompatibleProvider._content_to_text(message.get("content", ""))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"LLM response content is empty: {payload}")
        return content.strip()

    @staticmethod
    def _extract_text_from_responses(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()
        top_level_content = OpenAICompatibleProvider._content_to_text(payload.get("content"))
        if top_level_content:
            return top_level_content
        collected: list[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            content_text = OpenAICompatibleProvider._content_to_text(item.get("content"))
            if content_text:
                collected.append(content_text)
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())
            message_text = OpenAICompatibleProvider._content_to_text((item.get("message") or {}).get("content"))
            if message_text:
                collected.append(message_text)
        if collected:
            return "\n".join(collected)
        if payload.get("choices"):
            return OpenAICompatibleProvider._extract_text_from_chat_completions(payload)
        raise RuntimeError(f"Responses API returned no text output: {payload}")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                if item_type in {"", "text", "output_text", "message"}:
                    text_parts.append(text.strip())
                    continue
                if item_type == "input_text":
                    continue
            nested_text = OpenAICompatibleProvider._content_to_text(item.get("content"))
            if nested_text:
                text_parts.append(nested_text)
        return "\n".join(part for part in text_parts if part).strip()


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
    }


def _task_metadata(task: TaskSpec) -> dict[str, Any]:
    return {
        "benchmark_family": task.benchmark_family,
        "entry_kind": task.entry_kind,
        "level": task.level,
        "problem_id": task.problem_id,
        "backend": task.backend,
        "source_origin": task.source_origin,
        "semantic_profile": semantic_profile_to_prompt_dict(task.semantic_profile),
        "strategy_portfolio": strategy_portfolio_to_prompt_dict(task.strategy_portfolio),
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

def _used_strategy_names(context: AgentContext) -> list[str]:
    names: list[str] = []
    for snapshot in [context.current, *context.related, *context.leaders]:
        name = getattr(snapshot, "plan_strategy_name", None)
        if name and name not in names:
            names.append(name)
    return names


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    payload = parse_loose_json_dict(cleaned, allow_python_literal=True)
    if payload is None:
        raise json.JSONDecodeError("Unable to parse JSON object", cleaned, 0)
    return payload


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _task_prompt_suffix(task: TaskSpec, role: str) -> str:
    if task.benchmark_family == "kernelbench":
        backend_hint = _kernelbench_backend_hint(task, role)
        if role == "plan":
            base = (
                "\nThis task comes from the real KernelBench benchmark and is evaluated on CUDA. "
                "Return the requested planning JSON only and do not include markdown fences or prose."
            )
        elif role == "search":
            base = (
                "\nThis task comes from the real KernelBench benchmark and is evaluated on CUDA. "
                "Return the requested JSON object only; its code field must contain executable Python source without markdown fences or prose."
            )
        else:
            base = (
                "\nThis task comes from the real KernelBench benchmark and is evaluated on CUDA. "
                "Return executable Python code only and do not include markdown fences or prose."
            )
        anchor_hint = _kernelbench_anchor_hint(task)
        profile_hint = _kernelbench_profile_hint(task, role)
        if role == "search":
            return (
                base
                + " Generate a full candidate module directly, preserve the public evaluator I/O, keep ModelNew as the entry class when present, "
                + "and keep any existing scaffold structure stable unless the current code is already broken."
                + backend_hint
                + profile_hint
            )
        if role == "plan":
            return (
                base
                + f" Select the most relevant edit target from the provided structural anchors ({anchor_hint}). "
                + "Do not assume any handwritten adapter has pre-selected the target; infer the best region from the raw KernelBench ModelNew code and current feedback. "
                + "Preserve the generated scaffold and keep ModelNew as the entry class."
                + backend_hint
                + profile_hint
            )
        return (
            base
            + " The code must define ModelNew, keep the scaffold outside the anchors unchanged, preserve the __init__/forward signatures, and restrict edits to the provided anchors."
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
        if role == "plan":
            return (
                " Treat this as a native CUDA extension task: keep the scaffold intact, preserve the ModelNew interface, "
                "and propose grounded edits that stay inside helpers/cuda_cpp/cuda_cu/init_body/forward_stmt_*."
            )
        return (
            " Treat this as a native CUDA extension task: preserve the pybind binding surface, keep CUDA kernel launch assumptions explicit, "
            "and restrict edits to the grounded regions only."
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


def _normalize_patch_response(response_text: str, allowed_regions: dict[str, str]) -> str:
    cleaned = _strip_code_fences(str(response_text or "")).strip()
    payload = _parse_patch_payload_loose(cleaned)
    if payload is None:
        return cleaned
    normalized = _canonicalize_region_patches(payload, allowed_regions)
    if normalized is None:
        return cleaned
    return json.dumps({"region_patches": normalized}, ensure_ascii=False)


def _parse_patch_payload_loose(text: str) -> dict[str, Any] | None:
    return parse_loose_json_dict(text, allow_python_literal=True)


def _canonicalize_region_patches(payload: dict[str, Any], allowed_regions: dict[str, str]) -> list[dict[str, str]] | None:
    return canonicalize_region_patches(payload, allowed_regions)


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
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "http 524",
        "upstream_error",
        "origin_response_timeout",
        "timeout occurred",
        "temporarily unavailable",
        "timed out",
        "\"retryable\":true",
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "remote end closed connection",
        "connection reset by peer",
        "connection aborted",
        "connection closed",
        "ssl",
    )
    return any(token in message for token in retryable_tokens)


def _retry_delay_seconds(exc: Exception | None, attempt_index: int, default_backoff: float) -> float:
    delay = max(0.0, default_backoff * (attempt_index + 1))
    if exc is None:
        return delay
    message = str(exc)
    retry_after = _extract_retry_after_seconds(message)
    if retry_after is None:
        return delay
    return max(delay, retry_after)


def _extract_retry_after_seconds(message: str) -> float | None:
    patterns = (
        r'"retry_after"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r"retry-after[^0-9]*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


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
