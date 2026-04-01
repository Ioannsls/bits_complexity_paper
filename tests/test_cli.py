from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.dev import quality_gate, smoke_pipeline
from bits_complexity.experiments import run, run_family, run_parallel
from bits_complexity.plots import run_publication


class CliParsingTest(unittest.TestCase):
    def test_single_run_parser_exposes_choices_for_core_arguments(self) -> None:
        parser = run.build_parser()
        self.assertEqual(parser._option_string_actions["--method"].choices, ("ef21", "diana"))
        self.assertEqual(
            parser._option_string_actions["--compressor-pipeline"].choices,
            ("fp32", "db", "topk", "randk", "db_topk"),
        )
        self.assertEqual(
            parser._option_string_actions["--dynamic-mode"].choices,
            ("disabled", "static", "dynamic"),
        )

    def test_single_run_parser_accepts_quantizer_family(self) -> None:
        parser = run.build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "a9a",
                "--method",
                "diana",
                "--compressor-pipeline",
                "db",
                "--quantizer-family",
                "du",
            ]
        )
        self.assertEqual(args.quantizer_family, "du")

    def test_single_run_parser_accepts_auto_learning_rate_flag(self) -> None:
        parser = run.build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "a9a",
                "--method",
                "ef21",
                "--compressor-pipeline",
                "topk",
                "--auto-learning-rate",
            ]
        )
        self.assertTrue(args.auto_learning_rate)

    def test_single_run_parser_accepts_learning_rate_multiplier(self) -> None:
        parser = run.build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "a9a",
                "--method",
                "ef21",
                "--compressor-pipeline",
                "topk",
                "--learning-rate-multiplier",
                "2.0",
            ]
        )
        self.assertEqual(args.learning_rate_multiplier, 2.0)

    def test_family_parser_defaults_are_stable(self) -> None:
        parser = run_family.build_parser()
        args = parser.parse_args(["--dataset", "a9a", "--family", "db"])
        self.assertEqual(args.output_dir, Path("outputs"))
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.clients, 10)
        self.assertFalse(args.dry_run)

    def test_family_parser_accepts_auto_learning_rate_flag(self) -> None:
        parser = run_family.build_parser()
        args = parser.parse_args(["--dataset", "a9a", "--family", "db", "--auto-learning-rate"])
        self.assertTrue(args.auto_learning_rate)

    def test_family_parser_accepts_learning_rate_multiplier(self) -> None:
        parser = run_family.build_parser()
        args = parser.parse_args(
            ["--dataset", "a9a", "--family", "db", "--learning-rate-multiplier", "2.0"]
        )
        self.assertEqual(args.learning_rate_multiplier, 2.0)

    def test_family_parser_supports_rebuild_from_existing_flag(self) -> None:
        parser = run_family.build_parser()
        args = parser.parse_args(["--dataset", "a9a", "--family", "db", "--rebuild-from-existing"])
        self.assertTrue(args.rebuild_from_existing)

    def test_parallel_parser_accepts_config_argument(self) -> None:
        parser = run_parallel.build_parser()
        args = parser.parse_args(["--config", "custom.json"])
        self.assertEqual(args.config, Path("custom.json"))

    def test_parallel_parser_supports_plots_only_flag(self) -> None:
        parser = run_parallel.build_parser()
        args = parser.parse_args(["--plots-only"])
        self.assertTrue(args.plots_only)

    def test_parallel_parser_uses_runs_configs_as_default(self) -> None:
        parser = run_parallel.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, Path("runs") / "configs" / "full_run_parallel.json")

    def test_publication_parser_supports_manifest(self) -> None:
        parser = run_publication.build_parser()
        args = parser.parse_args(
            ["--dataset", "a9a", "--family", "db", "--manifest", "custom.json"]
        )
        self.assertEqual(args.manifest, Path("custom.json"))

    def test_smoke_parser_defaults_are_budget_aware(self) -> None:
        parser = smoke_pipeline.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.max_iterations, 3)
        self.assertEqual(args.cutoff_kbits_per_n, 0.05)

    def test_run_help_mentions_key_arguments_and_invariants(self) -> None:
        help_text = run.build_parser().format_help()
        self.assertIn("--compressor-pipeline", help_text)
        self.assertIn("--quantizer-family", help_text)
        self.assertIn("DB, так и DU", help_text)

    def test_quality_gate_help_mentions_coverage_threshold(self) -> None:
        help_text = run.build_parser().format_help()
        self.assertIn("--bits-per-value", help_text)
        quality_help = quality_gate.build_parser().format_help()
        self.assertIn("--coverage-threshold", quality_help)
        self.assertIn("quality gate", quality_help.lower())

    def test_run_publication_main_uses_manifest_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with patch("bits_complexity.plots.run_publication.plot_family_from_manifest") as mocked:
                mocked.return_value = Path(tmp_dir) / "plot.png"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "run_publication",
                        "--dataset",
                        "a9a",
                        "--family",
                        "db",
                        "--manifest",
                        str(manifest),
                    ],
                ):
                    run_publication.main()
            mocked.assert_called_once_with(manifest)


if __name__ == "__main__":
    unittest.main()
