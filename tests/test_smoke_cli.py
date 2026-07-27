import unittest

from stark.cli import _build_parser


class SmokeCliTests(unittest.TestCase):
    def test_kernelbench_defaults_to_main(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-kernelbench',
            '--level', '1',
            '--problem-id', '25',
            '--output-dir', 'runs/test_default_single',
        ])
        self.assertEqual(args.run_profile, 'main')

    def test_batch_parser_accepts_experiment(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-kernelbench-batch',
            '--experiment', 'quick',
            '--output-dir', 'runs/test_batch',
        ])
        self.assertEqual(args.command, 'run-kernelbench-batch')
        self.assertEqual(args.run_profile, 'quick')

    def test_single_parser_accepts_kernelbench_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-kernelbench',
            '--experiment', 'quick',
            '--level', '1',
            '--problem-id', '25',
            '--output-dir', 'runs/test_single',
        ])
        self.assertEqual(args.level, 1)
        self.assertEqual(args.problem_id, 25)

    def test_single_parser_accepts_tilelang_backend(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-kernelbench',
            '--level', '1',
            '--problem-id', '25',
            '--backend', 'tilelang',
            '--output-dir', 'runs/test_tilelang',
        ])
        self.assertEqual(args.backend, 'tilelang')

    def test_single_parser_accepts_cute_backend(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-kernelbench',
            '--level', '1',
            '--problem-id', '25',
            '--backend', 'cute',
            '--output-dir', 'runs/test_cute',
        ])
        self.assertEqual(args.backend, 'cute')

    def test_direct_kernelbench_parser_accepts_model_override(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-direct-kernelbench',
            '--level', '1',
            '--problem-id', '40',
            '--backend', 'cuda',
            '--provider', 'claude-compatible',
            '--model', 'claude-fable-5',
            '--output-dir', 'runs/test_direct',
        ])
        self.assertEqual(args.command, 'run-direct-kernelbench')
        self.assertEqual(args.provider, 'claude-compatible')
        self.assertEqual(args.model, 'claude-fable-5')

    def test_direct_kernelbench_batch_parser_accepts_manifest(self):
        parser = _build_parser()
        args = parser.parse_args([
            'run-direct-kernelbench-batch',
            '--experiment', 'quick',
            '--manifest', 'configs/tasks/main_l1_15.yaml',
            '--output-dir', 'runs/test_direct_batch',
        ])
        self.assertEqual(args.command, 'run-direct-kernelbench-batch')
        self.assertEqual(args.run_profile, 'quick')
