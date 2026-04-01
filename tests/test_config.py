from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.common.config import RunConfig, validate_config


class ConfigValidationTest(unittest.TestCase):
    def test_resolved_quantizer_family_defaults_follow_method_mapping(self) -> None:
        ef21 = RunConfig(dataset="a9a", method="ef21", compressor_pipeline="db")
        diana = RunConfig(dataset="a9a", method="diana", compressor_pipeline="db")
        topk = RunConfig(dataset="a9a", method="ef21", compressor_pipeline="topk")
        self.assertEqual(ef21.resolved_quantizer_family, "db")
        self.assertEqual(diana.resolved_quantizer_family, "du")
        self.assertEqual(topk.resolved_quantizer_family, "none")

    def test_ef21_du_mapping_is_allowed(self) -> None:
        validate_config(
            RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="db",
                quantizer_family="du",
            )
        )

    def test_invalid_diana_db_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "DIANA must use unbiased DU semantics"):
            validate_config(
                RunConfig(
                    dataset="a9a",
                    method="diana",
                    compressor_pipeline="db",
                    quantizer_family="db",
                )
            )

    def test_invalid_k_ratio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_ratio"):
            validate_config(
                RunConfig(
                    dataset="a9a",
                    method="ef21",
                    compressor_pipeline="topk",
                    k_ratio=0.0,
                )
            )

    def test_learning_rate_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_config(
                RunConfig(
                    dataset="a9a",
                    method="ef21",
                    compressor_pipeline="topk",
                    learning_rate=0.0,
                )
            )

    def test_run_slug_depends_on_learning_rate_mode_and_value(self) -> None:
        manual = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="topk",
            learning_rate=0.01,
        )
        auto = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="topk",
            learning_rate=0.01,
            learning_rate_auto=True,
        )
        self.assertNotEqual(manual.run_slug, auto.run_slug)
        self.assertNotEqual(
            manual.run_slug,
            RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="topk",
                learning_rate=0.02,
            ).run_slug,
        )


if __name__ == "__main__":
    unittest.main()
