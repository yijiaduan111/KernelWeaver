"""Local full-weight cudaLLM provider."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from typing import Any

from .openai_provider import OpenAICompatibleProvider, _compact_payload_text


@dataclass(slots=True)
class LocalCudaLLMConfig:
    """Resolved configuration for a local full-weight cudaLLM backend."""

    model_path: str
    device: str = "cuda:1"
    torch_dtype: str = "bfloat16"
    top_p: float = 0.95
    max_new_tokens: int = 4096
    timeout_seconds: int = 600
    plan_temperature: float = 0.7
    code_temperature: float = 0.6
    debug_temperature: float = 0.1
    reasoning_effort: str | None = None
    plan_reasoning_effort: str | None = None
    code_reasoning_effort: str | None = None
    debug_reasoning_effort: str | None = None
    max_retries: int = 1
    retry_backoff_seconds: float = 0.0
    trust_remote_code: bool = False
    use_chat_template: bool = True


class LocalCudaLLMProvider(OpenAICompatibleProvider):
    """本地 HF cudaLLM provider。

    这里复用 OpenAI provider 已经写好的 plan/code/debug prompt 结构，
    只替换最底层 `_chat(...)`，让 CodeAgent 可以直接走本地 full-weight 模型。
    """

    name = "local-cudallm"

    def __init__(self, config: LocalCudaLLMConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None

    def with_overrides(self, **overrides) -> "LocalCudaLLMProvider":
        return LocalCudaLLMProvider(replace(self.config, **overrides))

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _resolve_torch_dtype(torch, self.config.torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.device and self.config.device != "auto":
            model_kwargs["device_map"] = {"": self.config.device}
        else:
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs,
        )
        self._model.eval()
        self._torch = torch

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if self._torch is not None:
            try:
                for device_index in range(self._torch.cuda.device_count()):
                    with self._torch.cuda.device(device_index):
                        self._torch.cuda.empty_cache()
                self._torch.cuda.ipc_collect()
            except Exception:
                pass
            self._torch = None

    def _chat(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        del reasoning_effort
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        torch = self._torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _compact_payload_text(user_payload)},
        ]
        prompt_text = _compose_local_chat_prompt(
            tokenizer=self._tokenizer,
            messages=messages,
            use_chat_template=self.config.use_chat_template,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        first_device = next(self._model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        do_sample = float(temperature) > 0
        generation_kwargs["do_sample"] = do_sample
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(self.config.top_p)
        last_error: Exception | None = None
        attempts = max(1, int(self.config.max_retries))
        for attempt in range(attempts):
            try:
                with torch.no_grad():
                    output = self._model.generate(**inputs, **generation_kwargs)
                generated = output[0][inputs["input_ids"].shape[-1] :]
                text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
                if text:
                    return text
                raise RuntimeError("Local cudaLLM returned empty text output.")
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts and self.config.retry_backoff_seconds > 0:
                    time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"Local cudaLLM generation failed: {last_error}") from last_error
        raise RuntimeError("Local cudaLLM generation failed before receiving a response.")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass



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
