import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.experiment.resource_cost import http_attempt, normalize_usage, summarize_http
from src.providers.openai_provider import OpenAICompatibleConfig, OpenAICompatibleProvider


class ResourceCostTests(unittest.TestCase):
    def test_missing_usage_is_unknown(self):
        self.assertIsNone(normalize_usage({})["total_tokens"])
        self.assertIsNone(normalize_usage({"usage": {"prompt_tokens": True}})["input_tokens"])

    def test_chat_usage_does_not_double_count_cached_or_reasoning_tokens(self):
        usage = normalize_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 20}}})
        self.assertEqual(usage["total_tokens"], 130)
        self.assertEqual(usage["cached_input_tokens"], 80)

    def test_responses_usage(self):
        usage = normalize_usage({"usage": {"input_tokens": 20, "output_tokens": 10}})
        self.assertEqual(usage["total_tokens"], 30)

    def test_anthropic_usage_counts_cache_reads_and_writes(self):
        usage = normalize_usage({"usage": {"input_tokens": 20, "output_tokens": 10,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50}}, protocol="anthropic")
        self.assertEqual(usage["input_tokens"], 170)
        self.assertEqual(usage["total_tokens"], 180)
        self.assertEqual(usage["uncached_input_tokens"], 20)
        self.assertIsNone(normalize_usage({}, protocol="anthropic")["total_tokens"])

    def test_transport_failure_is_counted_and_no_credentials_logged(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="secret-test-value"))
        payload = {"usage": {"prompt_tokens": 20, "completion_tokens": 5},
                   "choices": [{"message": {"content": "private response"}}]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PYRAMID_COST_LOG": str(Path(tmp) / "cost.jsonl")}):
            with mock.patch.object(provider, "_post_json_unmetered", side_effect=[RuntimeError("private error"), payload]):
                with self.assertRaises(RuntimeError):
                    provider._post_json("https://example.invalid", {"model": "test", "messages": ["private prompt"]})
                provider._post_json("https://example.invalid", {"model": "test"})
            raw = (Path(tmp) / "cost.jsonl").read_text()
            summary = summarize_http([json.loads(line) for line in raw.splitlines()])
            self.assertEqual(summary["llm_requests_total"], 2)
            self.assertEqual(summary["llm_http_errors"], 1)
            self.assertEqual(summary["total_tokens_known_subtotal"], 25)
            self.assertIsNone(summary["total_tokens"])
            self.assertEqual(summary["total_tokens_missing_requests"], 1)
            self.assertIsNone(summary["api_cost_estimate"])
            for secret in ("secret-test-value", "private prompt", "private response", "private error"):
                self.assertNotIn(secret, raw)

    def test_unfinished_request_is_not_a_zero_cost_success(self):
        summary = summarize_http([{"event": "http_start", "request_id": "one"}])
        self.assertEqual(summary["llm_requests_unfinished"], 1)
        self.assertIsNone(summary["total_tokens"])

    def test_no_output_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("src.experiment.resource_cost._append") as writer:
            with http_attempt("test") as event:
                event["total_tokens"] = 30
            writer.assert_not_called()

    def test_claude_object_format_is_opt_in(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(
            api_key="test", model="claude-sonnet-4-6", claude_tool_choice_format="object"))
        with mock.patch.object(provider, "_post_json", return_value={}) as post:
            provider._chat_completions_request("system", {}, 0.0)
        self.assertEqual(post.call_args.args[1]["tool_choice"], {"type": "none"})

    def test_claude_format_env_validation(self):
        with mock.patch.dict(os.environ, {"MICU_API_KEY": "test", "MICU_CLAUDE_TOOL_CHOICE_FORMAT": "object"}):
            self.assertEqual(OpenAICompatibleProvider.from_env(env_prefix="MICU").config.claude_tool_choice_format, "object")
        with mock.patch.dict(os.environ, {"MICU_API_KEY": "test", "MICU_CLAUDE_TOOL_CHOICE_FORMAT": "invalid"}):
            with self.assertRaises(ValueError):
                OpenAICompatibleProvider.from_env(env_prefix="MICU")


if __name__ == "__main__":
    unittest.main()
