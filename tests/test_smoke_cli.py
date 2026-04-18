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
