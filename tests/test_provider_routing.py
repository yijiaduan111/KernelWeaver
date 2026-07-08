import argparse
import unittest

from stark.cli import _build_deliberation_runner, _resolve_provider_routing
from stark.models import StarkConfig


class ProviderRoutingTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            'command': 'run-kernelbench',
            'run_profile': 'main',
            'provider': None,
            'agent_provider_profile': None,
            'route_config': None,
            'plan_provider': None,
            'code_provider': None,
            'debug_provider': None,
            'search_provider': None,
            'search_profile': None,
            'evaluator_profile': None,
            'measurement_profile': None,
            'deliberation_profile': None,
            'task_config': None,
            'runtime_config': None,
            'env_file': None,
            'kernelbench_root': None,
            'workflow': None,
            'backend': None,
            'max_attempts': None,
            'epsilon': None,
            'verbose': False,
            'output_dir': 'runs/test',
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_codeagent_cudallm_route_is_resolved(self):
        args = self._args(agent_provider_profile='codeagent_cudallm')
        routing = _resolve_provider_routing(args, 'main')
        self.assertEqual(routing['plan_provider'], 'openai-compatible')
        self.assertEqual(routing['code_provider'], 'local-cudallm')
        self.assertEqual(routing['search_provider'], 'local-cudallm')

    def test_codeagent_claude_route_is_resolved(self):
        args = self._args(agent_provider_profile='codeagent_claude')
        routing = _resolve_provider_routing(args, 'main')
        self.assertEqual(routing['plan_provider'], 'openai-compatible')
        self.assertEqual(routing['code_provider'], 'claude-compatible')
        self.assertEqual(routing['search_provider'], 'claude-compatible')
    def test_codeagent_gemini_route_is_resolved(self):
        args = self._args(agent_provider_profile='codeagent_gemini')
        routing = _resolve_provider_routing(args, 'main')
        self.assertEqual(routing['plan_provider'], 'openai-compatible')
        self.assertEqual(routing['code_provider'], 'gemini-compatible')
        self.assertEqual(routing['search_provider'], 'gemini-compatible')


    def test_deliberation_runner_uses_deliberation_timeout_not_search_timeout(self):
        import stark.cli as cli

        captured = {}
        original_prepare = cli._prepare_runtime_and_env
        original_instantiate = cli._instantiate_single_provider

        def fake_prepare(args, run_name):
            del args, run_name

        def fake_instantiate(name, overrides=None):
            captured[name] = dict(overrides or {})
            from stark.providers import MockProvider

            return MockProvider()

        try:
            cli._prepare_runtime_and_env = fake_prepare
            cli._instantiate_single_provider = fake_instantiate
            args = self._args(search_profile='quick')
            config = StarkConfig(
                deliberation_enabled=True,
                deliberation_providers=['mock'],
                deliberation_provider_timeout_seconds=180,
                deliberation_phase_timeout_seconds=240,
            )
            runner = _build_deliberation_runner(args, 'main', config)
        finally:
            cli._prepare_runtime_and_env = original_prepare
            cli._instantiate_single_provider = original_instantiate

        self.assertIsNotNone(runner)
        self.assertEqual(captured['mock']['timeout_seconds'], 180)
        self.assertEqual(runner.phase_timeout_seconds, 240.0)

    def test_all_gemini_route_is_resolved(self):
        args = self._args(provider='gemini-compatible', agent_provider_profile='all_gemini')
        routing = _resolve_provider_routing(args, 'main')
        self.assertEqual(routing['plan_provider'], 'gemini-compatible')
        self.assertEqual(routing['code_provider'], 'gemini-compatible')
        self.assertEqual(routing['debug_provider'], 'gemini-compatible')


def test_http_522_is_retryable():
    from src.providers.openai_provider import _is_retryable_llm_error

    message = 'LLM request failed: HTTP 522: '
    assert _is_retryable_llm_error(RuntimeError(message))
