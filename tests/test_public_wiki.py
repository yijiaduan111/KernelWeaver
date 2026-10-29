"""Behavioral checks for identity independence and contamination isolation."""

from dataclasses import replace
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.knowledge import load_operator_memory
from src.knowledge.public_wiki import PUBLIC_ROOT, PublicKernelWiki, audit_public_corpus, operator_symbols
from src.models import StarkConfig, TaskSpec
from src.core.workflow import run_stark


def task(code, **kwargs):
    return TaskSpec(name="arbitrary", description="", source_code=code, reference_code=code,
                    function_name="f", reference_function_name="f", test_cases=[], benchmark_cases=[], backend="cuda", **kwargs)


class PublicWikiTests(unittest.TestCase):
    def setUp(self):
        self.wiki = PublicKernelWiki(PUBLIC_ROOT, target_arch="sm86")

    def test_identity_comments_fixture_shapes_and_outcomes_do_not_change_evidence(self):
        code = "import torch\ndef f(x): return torch.cumsum(x, dim=1)\ndef get_inputs(): return [torch.randn(7, 11)]"
        original = task(code, problem_id=89, level=1)
        changed_code = code.replace("(7, 11)", "(19, 23)") + "\n# torch.matmul torch.softmax P40 5.217x\n"
        changed = replace(original, name="L1P40 LayerNorm", description="best candidate 5.217x",
                          problem_id=40, tags=["layernorm"], reference_code=changed_code)
        self.assertEqual(self.wiki.retrieve(original), self.wiki.retrieve(changed))
        self.assertEqual(self.wiki.retrieve(original)[0]["id"], "public-scan")

    def test_import_aliases_module_constructors_and_at_operator(self):
        examples = [("import torch.nn.functional as op\ndef f(x): return op.cross_entropy(x, y)", "public-cross_entropy"),
                    ("from torch.nn import LayerNorm as N\nclass Renamed:\n def __init__(self): self.layer=N((7,11))", "public-layernorm"),
                    ("def f(a,b): return a @ b", "public-matmul"),
                    ("def f(x): return x.sum(dim=-1)", "public-reduction")]
        for code, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(self.wiki.retrieve(task(code))[0]["id"], expected)

    def test_scan_and_transposed_convolution_do_not_get_generic_namesake(self):
        scan = self.wiki.retrieve(task("import torch\ndef f(x): return torch.cumsum(x, 0)"))
        self.assertNotIn("public-reduction", [p["id"] for p in scan])
        transposed = self.wiki.retrieve(task("from torch import nn\nlayer=nn.ConvTranspose3d(3, 7, 3)"))
        self.assertEqual(transposed[0]["id"], "public-transposed_convolution")
        self.assertNotIn("public-convolution", [p["id"] for p in transposed])

    def test_unknown_is_explicit_fallback_and_other_backend_receives_nothing(self):
        source = task("def f(x): return custom_unknown_op(x)")
        self.assertEqual(self.wiki.retrieve(source)[0]["coverage"], "general_contract_only")
        self.assertEqual(self.wiki.retrieve(replace(source, backend="npu")), [])
        with self.assertRaisesRegex(ValueError, "does not support"):
            PublicKernelWiki(PUBLIC_ROOT, target_arch="sm100")

    def test_legacy_and_experimental_pages_cannot_enter_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp)
            (legacy / "index.yaml").write_text("knowledge_scope: task_adapted\n")
            with self.assertRaisesRegex(ValueError, "knowledge_scope"):
                load_operator_memory(legacy, target_arch="sm86", public_only=True)
        with self.assertRaisesRegex(ValueError, "experimental"):
            PublicKernelWiki(PUBLIC_ROOT, target_arch="sm86", experimental_page_ids=["sm86-maxpool2d-window"])

    def test_modified_corpus_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "public"
            shutil.copytree(PUBLIC_ROOT, path)
            page = path / "wiki/operators/matmul.md"
            page.write_text(page.read_text() + "\nChanged guidance.\n")
            with self.assertRaisesRegex(ValueError, "changed since freeze"):
                PublicKernelWiki(path, target_arch="sm86")

    def test_leaking_and_unattributed_pages_rejected_even_before_freeze(self):
        for before, after, message in [
            ("Inferred optimization", "P40 candidate optimization", "marker"),
            ('"public_documentation"', '"task_experiment"', "empirical"),
            ('"source-torch-ops"', '"source-nonexistent"', "Missing primary source"),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "public"
                shutil.copytree(PUBLIC_ROOT, path)
                page = path / "wiki/operators/matmul.md"
                page.write_text(page.read_text().replace(before, after))
                with self.assertRaisesRegex(ValueError, message):
                    audit_public_corpus(path, check_lock=False)

    def test_strict_workflow_rejects_experience_before_evaluation(self):
        evaluator = Mock()
        config = StarkConfig(operator_memory_public_only=True, long_term_memory_enabled=True)
        with self.assertRaisesRegex(ValueError, "cross-task"):
            run_stark(task("pass"), config, Mock(), evaluator)
        evaluator.evaluate.assert_not_called()

    def test_strict_config_is_wired_through_cli(self):
        from src.cli import _build_config, _build_parser
        args = _build_parser().parse_args(["run-kernelbench", "--level", "1", "--problem-id", "1",
                                          "--search-profile", "main_memory_public_wiki_sm86", "--output-dir", "runs/test_public"])
        config = _build_config(args, "main")
        self.assertTrue(config.operator_memory_public_only)
        self.assertTrue(config.operator_memory_enabled)
        self.assertFalse(config.long_term_memory_enabled)

    def test_all_development_tasks_have_specialized_pages_without_identity_rules(self):
        from src.core.loader import KernelBenchLoader
        from src.experiment.batch_runner import load_task_manifest
        root = PUBLIC_ROOT.parents[1]
        manifest = load_task_manifest(root / "configs/tasks/main_l1_15.yaml")
        for entry in manifest["tasks"]:
            original = KernelBenchLoader().load_official_problem(root / "KernelBench", entry["level"], entry["problem_id"], backend="cuda")
            anonymized = replace(original, name="opaque", description="", problem_id=None, level=None, tags=[], semantic_profile=None)
            with self.subTest(problem_id=entry["problem_id"]):
                evidence = self.wiki.retrieve(anonymized)
                self.assertTrue(any(p["page_type"] == "kernel" for p in evidence))
                self.assertEqual(evidence, self.wiki.retrieve(original))
                for page in evidence:
                    self.assertEqual(page["knowledge_scope"], "public_general")
                    self.assertTrue(page["source_records"])

    def test_matched_control_differs_only_in_wiki_configuration(self):
        import yaml
        root = PUBLIC_ROOT.parents[1] / "configs/search"
        control = yaml.safe_load((root / "main_memory_public_control.yaml").read_text())
        treatment = yaml.safe_load((root / "main_memory_public_wiki_sm86.yaml").read_text())
        shared = lambda config: {k: v for k, v in config.items()
                                 if k != "description" and not k.startswith("operator_memory_")}
        self.assertEqual(shared(control), shared(treatment))
        self.assertFalse(control["long_term_memory_enabled"])
        self.assertFalse(control["operator_memory_enabled"])
        self.assertTrue(treatment["operator_memory_enabled"])
