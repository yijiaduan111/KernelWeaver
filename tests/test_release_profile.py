from dataclasses import replace
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from src.cli import _build_config, _build_parser
from src.core.workflow import run_stark
from src.demo import build_demo_tasks
from src.evaluation import DemoEvaluator
from src.providers import MockProvider
from src.models import StarkConfig


ROOT = Path(__file__).resolve().parents[1]


class RecordingProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def propose_plan(self, task, node, context):
        self.contexts.append(context)
        return super().propose_plan(task, node, context)


class ReleaseProfileTests(unittest.TestCase):
    def test_main_defaults_to_memory_only_without_architecture_scoped_wiki(self):
        args = _build_parser().parse_args(['run-kernelbench','--level','1','--problem-id','25',
                                          '--output-dir','runs/test_default_memory'])
        config = _build_config(args,'main')
        self.assertEqual(config.search_profile,'main_memory_longterm')
        self.assertTrue(config.long_term_memory_enabled)
        self.assertGreater(config.context_limit,0)
        self.assertFalse(config.operator_memory_enabled)
        self.assertIsNone(config.operator_memory_path)

    def test_wiki_opt_in_is_rejected_before_evaluation(self):
        evaluator = Mock()
        for options in ({'operator_memory_enabled':True}, {'domain_wiki_enabled':True},
                        {'operator_memory_public_corpus':True}):
            with self.assertRaisesRegex(ValueError,'Wiki is unavailable'):
                run_stark(build_demo_tasks()[0],StarkConfig(**options),MockProvider(),evaluator)
        evaluator.evaluate.assert_not_called()

    def test_no_wiki_modules_profiles_or_corpus_are_shipped(self):
        self.assertFalse((ROOT/'src/knowledge').exists())
        self.assertFalse((ROOT/'knowledge').exists())
        self.assertFalse(list((ROOT/'configs/search').glob('*wiki*')))

    def test_memory_only_profile_uses_short_term_context_and_fresh_persistent_journal(self):
        with TemporaryDirectory() as tmp:
            args = _build_parser().parse_args(['run-demo','--task','square_list','--experiment','quick',
                '--provider','mock','--route-config','all_mock','--search-config','main_memory_longterm',
                '--max-attempts','2','--output-dir',tmp])
            config = _build_config(args,'quick')
            self.assertTrue(config.long_term_memory_enabled)
            self.assertFalse(config.operator_memory_public_corpus)
            self.assertFalse(config.operator_memory_enabled)
            self.assertGreater(config.context_limit,0)
            self.assertFalse(config.operator_memory_public_only)
            self.assertFalse(config.profile_feedback_enabled)
            task = replace(build_demo_tasks()[0], backend='cuda')
            first = RecordingProvider()
            self.assertFalse(Path(config.long_term_memory_path).exists())
            run_stark(task,config,first,DemoEvaluator())
            self.assertTrue(Path(config.long_term_memory_path).exists())
            self.assertTrue(first.contexts)
            self.assertEqual(first.contexts[0].long_term,[])
            self.assertTrue(all(not c.operator_memory for c in first.contexts))
            second = RecordingProvider()
            run_stark(task,config,second,DemoEvaluator())
            self.assertTrue(second.contexts[0].long_term)
            self.assertTrue(all(not c.operator_memory for c in second.contexts))

    def test_generated_launch_script_does_not_persist_credentials(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'launch.sh'
            script = 'source scripts/common.sh\ncommand=(python -V)\nkw_write_launch_script "$1" "$PWD" command\n'
            env = dict(os.environ,OPENAI_API_KEY='test-private-openai-credential',
                       CLAUDE_API_KEY='test-private-claude-credential')
            subprocess.run(['bash','-c',script,'test',str(output)],cwd=ROOT,env=env,check=True)
            text = output.read_text()
            self.assertNotIn('test-private-',text)
            self.assertNotIn('export OPENAI_API_KEY=',text)
            self.assertNotIn('export CLAUDE_API_KEY=',text)


if __name__=='__main__':
    unittest.main()
