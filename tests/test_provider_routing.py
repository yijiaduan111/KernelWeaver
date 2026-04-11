import argparse
import unittest

from stark.cli import _resolve_provider_routing


class ProviderRoutingTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            'command': 'run-kernelbench',
            'run_profile': 'paper_mini',
            'preset': None,
            'provider': None,
            'agent_provider_profile': None,
            'plan_provider': None,
            'code_provider': None,
            'debug_provider': None,
            'search_provider': None,
            'search_profile': None,
            'evaluator_profile': None,
            'measurement_profile': None,
            'evaluation_profile': None,
            'kernelbench_evaluator': None,
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
        routing = _resolve_provider_routing(args, 'paper_mini')
        self.assertEqual(routing['plan_provider'], 'openai-compatible')
        self.assertEqual(routing['code_provider'], 'local-cudallm')
        self.assertEqual(routing['search_provider'], 'local-cudallm')
