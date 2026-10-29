import argparse
import os
import unittest
from unittest import mock

from src.config import provider_defaults
from src.providers import OpenAICompatibleProvider
from stark.cli import _resolve_provider_routing


class ProviderRoutingTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            'command': 'run-kernelbench',
            'run_profile': 'main',
            'provider': None,
            'agent_provider_profile': None,
            'plan_provider': None,
            'code_provider': None,
            'debug_provider': None,
            'search_provider': None,
            'search_profile': None,
            'evaluator_profile': None,
            'measurement_profile': None,
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

    def test_all_micu_route_is_resolved(self):
        args = self._args(agent_provider_profile='all_micu', provider='micu-compatible')
        routing = _resolve_provider_routing(args, 'main')
        self.assertEqual(routing['plan_provider'], 'micu-compatible')
        self.assertEqual(routing['code_provider'], 'micu-compatible')
        self.assertEqual(routing['debug_provider'], 'micu-compatible')
        self.assertEqual(routing['search_provider'], 'micu-compatible')

    def test_micu_provider_uses_its_own_environment_prefix(self):
        with mock.patch.dict(os.environ, {'MICU_API_KEY': 'test-key'}, clear=True):
            provider = OpenAICompatibleProvider.from_env(
                provider_defaults('micu-compatible'),
                env_prefix='MICU',
            )
        self.assertEqual(provider.config.base_url, 'https://www.micuapi.ai/v1')
        self.assertEqual(provider.config.model, 'claude-sonnet-4-5-20250929')
        self.assertEqual(provider.config.wire_api, 'chat_completions')
        self.assertEqual(provider.config.max_tokens, 8192)
        self.assertEqual(provider._build_endpoint('chat_completions'), 'https://www.micuapi.ai/v1/chat/completions')
