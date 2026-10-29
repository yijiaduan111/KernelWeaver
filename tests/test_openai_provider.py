import http.client
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from src.providers.openai_provider import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    _parse_json_object,
    _task_prompt_suffix,
)
from src.models import AgentContext, NodeSnapshot, SearchNode, TaskSpec


class OpenAIProviderRobustnessTests(unittest.TestCase):
    def test_json_correction_does_not_retry_transport_errors(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="test"))
        provider._chat = mock.Mock(side_effect=RuntimeError("HTTP 400: invalid request"))
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            provider._chat_json_object("system", {}, 0, required_keys=("value",))
        self.assertEqual(provider._chat.call_count, 1)

    def test_operator_memory_is_only_sent_when_exact_evidence_exists(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="test"))
        provider._chat_json_object = mock.Mock(return_value={
            "strategy_name": "grid_stride",
            "strategy_summary": "apply the bounded SM86 hypothesis",
            "expected_gain": "lower measured runtime",
            "risk_notes": "tail alignment",
            "anchor_edits": [{"anchor_name": "forward_stmt_1", "instruction": "change mapping"}],
        })
        task = self._kernelbench_task()
        node = SearchNode("root", None, 0, task.source_code, "root")
        snap = NodeSnapshot("root", None, 0, None, "correct", None, None, 0, "root", 0, None, None)
        evidence = [{"id": "sm86-elementwise-single-pass", "avoid": ["no new build path"]}]
        provider.propose_plan(task, node, AgentContext("plan", snap, snap, operator_memory=evidence))
        call = provider._chat_json_object.call_args
        self.assertEqual(call.kwargs["user_payload"]["long_term_operator_memory"], evidence)
        self.assertIn("curated, architecture-scoped prior", call.kwargs["system_prompt"])
        provider._chat_json_object.reset_mock()
        provider.propose_plan(task, node, AgentContext("plan", snap, snap))
        self.assertNotIn("long_term_operator_memory", provider._chat_json_object.call_args.kwargs["user_payload"])

    def test_kda_guidance_is_explicit_and_only_sent_when_enabled(self):
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="test"))
        provider._chat_json_object = mock.Mock(return_value={
            "strategy_name": "coalescing",
            "strategy_summary": "measured evidence -> transaction hypothesis",
            "expected_gain": "lower measured runtime",
            "risk_notes": "alignment",
            "anchor_edits": [{"anchor_name": "forward_stmt_1", "instruction": "change mapping"}],
        })
        task = self._kernelbench_task()
        node = SearchNode("root", None, 0, task.source_code, "root")
        snap = NodeSnapshot("root", None, 0, None, "correct", None, None, 0, "root", 0, None, None)
        context = AgentContext("plan", snap, snap, workflow_guidance={"method": "kda_style_evidence_loop"})
        provider.propose_plan(task, node, context)
        call = provider._chat_json_object.call_args
        self.assertIn("exactly one primary performance hypothesis", call.kwargs["system_prompt"])
        self.assertEqual(call.kwargs["user_payload"]["workflow_guidance"]["method"], "kda_style_evidence_loop")

    def test_paired_cache_replays_no_memory_calls_in_recorded_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = str(Path(tmp) / "paired.json")
            recorder = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test", paired_cache_path=cache_path, paired_cache_mode="record")
            )
            recorder._chat_completions_request = mock.Mock(
                side_effect=[
                    {"choices": [{"message": {"content": "first"}}]},
                    {"choices": [{"message": {"content": "second"}}]},
                ]
            )
            self.assertEqual(recorder._chat("system", {"current_node": {"runtime": 1.0}}, 0.0), "first")
            self.assertEqual(recorder._chat("system", {"current_node": {"runtime": 1.0}}, 0.0), "second")

            replay = OpenAICompatibleProvider(
                OpenAICompatibleConfig(api_key="test", paired_cache_path=cache_path, paired_cache_mode="replay")
            )
            replay._chat_completions_request = mock.Mock(side_effect=AssertionError("network should not be called"))
            self.assertEqual(replay._chat("system", {"current_node": {"runtime": 9.0}}, 0.0), "first")
            self.assertEqual(replay._chat("system", {"current_node": {"runtime": 9.0}}, 0.0), "second")

    def test_paired_cache_never_replays_visible_cross_candidate_memory(self):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test", paired_cache_path="unused.json", paired_cache_mode="replay")
        )
        provider._chat_completions_request = mock.Mock(return_value={"choices": [{"message": {"content": "live"}}]})
        result = provider._chat("system", {"related_nodes": [{"node_id": "n1"}]}, 0.0)
        self.assertEqual(result, "live")
        self.assertEqual(provider._chat_completions_request.call_count, 1)

    @staticmethod
    def _kernelbench_task():
        return TaskSpec(
            name="demo",
            description="demo",
            source_code="# <<<IMPROVE:forward_stmt_1>>>\nreturn x\n# <<<END_IMPROVE>>>",
            reference_code="return x",
            function_name="ModelNew",
            reference_function_name="Model",
            test_cases=[],
            benchmark_cases=[],
            benchmark_family="kernelbench",
            backend="cuda",
        )

    def test_parse_json_object_accepts_markdown_prose(self):
        payload = _parse_json_object('Here is the result:\n```json\n{"value": 7}\n```')
        self.assertEqual(payload, {"value": 7})

    def test_chat_json_object_retries_invalid_model_text(self):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test", max_retries=2, retry_backoff_seconds=0)
        )
        provider._chat = mock.Mock(side_effect=["not json", '{"strategy_name":"ok"}'])
        payload = provider._chat_json_object(
            "system",
            {"task": "demo"},
            0.1,
            required_keys=("strategy_name",),
        )
        self.assertEqual(payload["strategy_name"], "ok")
        self.assertEqual(provider._chat.call_count, 2)
        retry_call = provider._chat.call_args_list[1]
        self.assertIn("CORRECTION REQUIRED", retry_call.args[0])
        self.assertEqual(retry_call.args[1]["previous_invalid_response"], "not json")
        self.assertEqual(retry_call.args[2], 0.0)

    def test_kernelbench_plan_suffix_does_not_request_python_code(self):
        suffix = _task_prompt_suffix(self._kernelbench_task(), "plan")
        self.assertIn("planning schema", suffix)
        self.assertNotIn("Return executable Python code only", suffix)

    def test_kernelbench_code_suffix_requests_python_code(self):
        suffix = _task_prompt_suffix(self._kernelbench_task(), "code")
        self.assertIn("Return executable Python code only", suffix)

    def test_micu_claude_chat_uses_top_level_system_prompt(self):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test", model="claude-sonnet-4-6")
        )
        provider._post_json = mock.Mock(return_value={"choices": []})
        provider._chat_completions_request("system text", {"task": "demo"}, 0.1)
        body = provider._post_json.call_args.args[1]
        self.assertEqual(body["system"], "system text")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["tool_choice"], "none")
        self.assertNotIn("thinking", body)
        self.assertEqual([message["role"] for message in body["messages"]], ["user"])

    def test_non_claude_chat_keeps_system_message(self):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test", model="gpt-5.4")
        )
        provider._post_json = mock.Mock(return_value={"choices": []})
        provider._chat_completions_request("system text", {"task": "demo"}, 0.1)
        body = provider._post_json.call_args.args[1]
        self.assertNotIn("system", body)
        self.assertNotIn("tool_choice", body)
        self.assertEqual([message["role"] for message in body["messages"]], ["system", "user"])

    @mock.patch("src.providers.openai_provider.urllib.request.urlopen")
    def test_remote_disconnect_is_wrapped_for_retry(self, urlopen):
        urlopen.side_effect = http.client.RemoteDisconnected("remote closed")
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="test"))
        with self.assertRaisesRegex(RuntimeError, "remote closed"):
            provider._post_json("https://example.invalid/v1/chat/completions", {"model": "demo"})

    def test_chat_retries_empty_gateway_completion(self):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test", max_retries=2, retry_backoff_seconds=0)
        )
        provider._chat_completions_request = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
                {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
            ]
        )
        self.assertEqual(provider._chat("system", {}, 0.1), "OK")
        self.assertEqual(provider._chat_completions_request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
