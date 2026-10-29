import json
import io
import http.client
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from stark.providers import ClaudeCompatibleConfig, ClaudeCompatibleProvider


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ClaudeProviderTests(unittest.TestCase):
    @staticmethod
    def _stream(events):
        return io.BytesIO("".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode())

    def test_stream_joins_text_and_uses_cumulative_usage(self):
        payload = ClaudeCompatibleProvider._read_message_stream(self._stream([
            {"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "he"}},
            {"type": "ping"},
            {"type": "future_event"},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "llo"}},
            {"type": "message_delta", "usage": {"output_tokens": 8}, "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]))
        self.assertEqual(payload["content"][0]["text"], "hello")
        self.assertEqual(payload["usage"], {"input_tokens": 20, "output_tokens": 8})

    def test_stream_truncation_is_not_a_success(self):
        with self.assertRaisesRegex(RuntimeError, "before message_stop"):
            ClaudeCompatibleProvider._read_message_stream(self._stream([
                {"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}},
            ]))

    def test_remote_disconnect_enters_existing_retry_loop(self):
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test-key", retry_backoff_seconds=0))
        with patch("urllib.request.urlopen", side_effect=[http.client.RemoteDisconnected("closed"),
                    _DummyResponse({"content": [{"type": "text", "text": "ok"}]})]) as mocked:
            self.assertEqual(provider._chat("test", {"hello": "world"}, 0), "ok")
            self.assertEqual(mocked.call_count, 2)

    def test_messages_endpoint_uses_v1_messages(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://open.xiaojingai.com",
                model="claude-sonnet-4-6",
            )
        )
        self.assertEqual(provider._messages_endpoint(), "https://open.xiaojingai.com/v1/messages")

    def test_post_json_adds_anthropic_version_header(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://open.xiaojingai.com",
                model="claude-sonnet-4-6",
                api_version="2023-06-01",
            )
        )
        captured = {}

        def _fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _DummyResponse({"content": [{"text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            payload = provider._post_json(
                provider._messages_endpoint(),
                {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hello"}]},
            )

        headers = {key.lower(): value for key, value in captured["request"].header_items()}
        self.assertEqual(payload["content"][0]["text"], "ok")
        self.assertEqual(captured["timeout"], provider.config.timeout_seconds)
        self.assertEqual(headers["authorization"], "Bearer test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(json.loads(captured["request"].data)["tool_choice"], {"type": "none"})

    def test_tool_reply_is_retried_and_all_usage_is_counted(self):
        from stark.experiment.resource_cost import summarize_http
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test", retry_backoff_seconds=0))
        tool = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "NEVER_EXECUTE"}}],
                "usage": {"input_tokens": 10, "output_tokens": 2}}
        good = {"content": [{"type": "text", "text": "OK"}], "usage": {"input_tokens": 12, "output_tokens": 3}}
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp) / "events.jsonl"
            with patch.dict(os.environ, {"PYRAMID_COST_LOG": str(logfile), "CLAUDE_STREAM": "0"}), \
                 patch("urllib.request.urlopen", side_effect=[_DummyResponse(tool), _DummyResponse(good)]) as request:
                self.assertEqual(provider._chat("test", {"task": "hello"}, 0), "OK")
            totals = summarize_http([json.loads(s) for s in logfile.read_text().splitlines()])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(totals["llm_retries"], 1)
            self.assertEqual(totals["total_tokens"], 27)

    def test_tool_retry_is_bounded_and_partial_text_is_not_accepted(self):
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test", max_retries=2, retry_backoff_seconds=0))
        payload = {"content": [{"type": "text", "text": "I will use bash"}, {"type": "tool_use", "name": "Bash"}]}
        with patch.object(provider, "_messages_request", return_value=payload) as request:
            with self.assertRaisesRegex(RuntimeError, "unexpected tool_use"):
                provider._chat("test", {}, 0)
        self.assertEqual(request.call_count, 2)

    def test_empty_text_is_retried(self):
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test", retry_backoff_seconds=0))
        with patch.object(provider, "_messages_request", side_effect=[{"content": []}, {"content": [{"text": "OK"}]}]):
            self.assertEqual(provider._chat("test", {}, 0), "OK")

    def test_opt_in_tool_recovery_returns_errors_never_executes_tools(self):
        from stark.experiment.resource_cost import summarize_http
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test", retry_backoff_seconds=0))
        tool = {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "tool_a", "name": "Bash", "input": {"command": "NEVER_EXECUTE"}},
            {"type": "tool_use", "id": "tool_b", "name": "Read", "input": {"path": "NEVER_READ"}}],
            "usage": {"input_tokens": 10, "output_tokens": 2}}
        good = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "FINAL"}],
                "usage": {"input_tokens": 12, "output_tokens": 3}}
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp) / "events.jsonl"
            with patch.dict(os.environ, {"PYRAMID_COST_LOG": str(logfile), "CLAUDE_TOOL_RECOVERY": "1", "CLAUDE_STREAM": "0"}), \
                 patch("urllib.request.urlopen", side_effect=[_DummyResponse(tool), _DummyResponse(good)]) as request:
                self.assertEqual(provider._chat("test", {"source": "supplied"}, 0), "FINAL")
            body = json.loads(request.call_args_list[1].args[0].data)
            self.assertEqual(body["tools"], [])
            self.assertEqual(body["tool_choice"], {"type": "none"})
            self.assertEqual(body["messages"][1], {"role": "assistant", "content": tool["content"]})
            errors = body["messages"][2]["content"]
            self.assertEqual([item["tool_use_id"] for item in errors], ["tool_a", "tool_b"])
            self.assertTrue(all(item["is_error"] for item in errors))
            totals = summarize_http([json.loads(s) for s in logfile.read_text().splitlines()])
            self.assertEqual(totals["llm_requests_total"], 2)
            self.assertEqual(totals["total_tokens"], 27)

    def test_opt_in_repeated_tool_calls_still_respect_existing_request_limit(self):
        provider = ClaudeCompatibleProvider(ClaudeCompatibleConfig(api_key="test", max_retries=2, retry_backoff_seconds=0))
        tool = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "tool_a", "name": "Bash", "input": {}}]}
        with patch.dict(os.environ, {"CLAUDE_TOOL_RECOVERY": "1"}), \
             patch.object(provider, "_messages_request", return_value=tool) as request:
            with self.assertRaisesRegex(RuntimeError, "unexpected tool_use"):
                provider._chat("test", {}, 0)
            self.assertEqual(request.call_count, 2)

    def test_malformed_tool_reply_cannot_create_an_error_conversation(self):
        for content in ([{"type": "tool_use"}], [{"type": "server_tool_use", "id": "s"}],
                        [{"type": "tool_use", "id": "a"}, {"type": "tool_use", "id": "a"}]):
            self.assertIsNone(ClaudeCompatibleProvider._denied_tool_history({"content": content}, None, {}))

    def test_stream_preserves_tool_arguments_and_thinking_signatures(self):
        payload = ClaudeCompatibleProvider._read_message_stream(self._stream([
            {"type": "message_start", "message": {"usage": {}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "thinking"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "a", "name": "Bash", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"command\":"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"NEVER_EXECUTE\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]))
        self.assertEqual(payload["content"][0]["signature"], "sig")
        self.assertEqual(payload["content"][1]["input"], {"command": "NEVER_EXECUTE"})


if __name__ == "__main__":
    unittest.main()
