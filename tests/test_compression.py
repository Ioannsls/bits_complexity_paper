from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.pipelines import build_pipeline


class CompressionPipelineTest(unittest.TestCase):
    def test_fp32_accounting_is_dense(self) -> None:
        config = RunConfig(
            dataset="a9a", method="ef21", compressor_pipeline="fp32", bits_per_value=32
        )
        pipeline = build_pipeline(config=config, dimension=10)
        result = pipeline.compress(np.arange(10, dtype=np.float64))
        self.assertEqual(result.stats.transmitted_coordinates, 10)
        self.assertEqual(result.stats.bits_for_values, 320)
        self.assertEqual(result.stats.bits_for_indices, 0)

    def test_topk_accounting_tracks_sparse_indices(self) -> None:
        config = RunConfig(dataset="a9a", method="ef21", compressor_pipeline="topk", k_ratio=0.1)
        pipeline = build_pipeline(config=config, dimension=10)
        result = pipeline.compress(np.arange(10, dtype=np.float64))
        self.assertEqual(result.stats.transmitted_coordinates, 1)
        self.assertEqual(result.stats.bits_for_values, 32)
        self.assertEqual(result.stats.bits_for_indices, 4)
        self.assertEqual(np.count_nonzero(result.vector), 1)

    def test_db_topk_uses_quantized_value_bits(self) -> None:
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db_topk",
            dynamic_mode="static",
            bits_per_value=4,
            k_ratio=0.2,
        )
        pipeline = build_pipeline(config=config, dimension=10)
        pipeline.prepare_round([np.linspace(-1.0, 1.0, num=10)])
        result = pipeline.compress(np.linspace(-1.0, 1.0, num=10))
        self.assertEqual(result.stats.transmitted_coordinates, 2)
        self.assertEqual(result.stats.bits_for_values, 8)
        self.assertEqual(result.stats.bits_for_indices, 8)
        self.assertEqual(np.count_nonzero(result.vector), 2)

    def test_randk_keeps_exact_sparse_budget(self) -> None:
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="randk",
            quantizer_family="none",
            k_ratio=0.2,
            seed=7,
        )
        pipeline = build_pipeline(config=config, dimension=10)
        result = pipeline.compress(np.linspace(-1.0, 1.0, num=10))
        self.assertEqual(result.stats.transmitted_coordinates, 2)
        self.assertEqual(np.count_nonzero(result.vector), 2)

    def test_db_and_du_have_distinct_semantics(self) -> None:
        prepare_vector = np.full(64, 0.8, dtype=np.float64)
        vector = np.full(64, 0.45, dtype=np.float64)
        db_config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="static",
            bits_per_value=4,
            seed=1,
        )
        du_config = RunConfig(
            dataset="a9a",
            method="diana",
            compressor_pipeline="db",
            quantizer_family="du",
            dynamic_mode="static",
            bits_per_value=4,
            seed=1,
        )
        db_pipeline = build_pipeline(config=db_config, dimension=64)
        du_pipeline = build_pipeline(config=du_config, dimension=64)
        db_pipeline.prepare_round([prepare_vector])
        du_pipeline.prepare_round([prepare_vector])
        db_result = db_pipeline.compress(vector)
        du_result = du_pipeline.compress(vector)
        self.assertEqual(db_config.resolved_quantizer_family, "db")
        self.assertEqual(du_config.resolved_quantizer_family, "du")
        self.assertGreater(np.count_nonzero(np.abs(du_result.vector - db_result.vector) > 0), 0)

    def test_dynamic_round_rule_updates_shared_lattice_only_after_factor_p_drop(self) -> None:
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="dynamic",
            bits_per_value=4,
            p=2,
        )
        pipeline = build_pipeline(config=config, dimension=3)
        pipeline.prepare_round([np.array([4.0, 0.0, 0.0])])
        self.assertAlmostEqual(pipeline._current_amax, 4.0)
        initial_levels = pipeline._levels.copy()
        pipeline.prepare_round([np.array([3.0, 0.0, 0.0])])
        self.assertAlmostEqual(pipeline._current_amax, 4.0)
        self.assertTrue(np.array_equal(pipeline._levels, initial_levels))
        pipeline.prepare_round([np.array([2.0, 0.0, 0.0])])
        self.assertAlmostEqual(pipeline._current_amax, 2.0)
        self.assertFalse(np.array_equal(pipeline._levels, initial_levels))

    def test_static_round_rule_freezes_shared_lattice_after_initialization(self) -> None:
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="static",
            bits_per_value=4,
            p=2,
        )
        pipeline = build_pipeline(config=config, dimension=3)
        pipeline.prepare_round([np.array([4.0, 0.0, 0.0])])
        initial_amax = pipeline._current_amax
        initial_levels = pipeline._levels.copy()

        pipeline.prepare_round([np.array([8.0, 0.0, 0.0])])
        self.assertAlmostEqual(pipeline._current_amax, initial_amax)
        self.assertTrue(np.array_equal(pipeline._levels, initial_levels))

        pipeline.prepare_round([np.array([1.0, 0.0, 0.0])])
        self.assertAlmostEqual(pipeline._current_amax, initial_amax)
        self.assertTrue(np.array_equal(pipeline._levels, initial_levels))

    def test_static_and_dynamic_compress_identically_for_same_lattice_state(self) -> None:
        static_config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="static",
            bits_per_value=4,
            p=2,
        )
        dynamic_config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="dynamic",
            bits_per_value=4,
            p=2,
        )
        static_pipeline = build_pipeline(config=static_config, dimension=5)
        dynamic_pipeline = build_pipeline(config=dynamic_config, dimension=5)
        round_vectors = [np.array([4.0, -2.0, 1.0, 0.5, 0.0])]
        static_pipeline.prepare_round(round_vectors)
        dynamic_pipeline.prepare_round(round_vectors)

        vector = np.array([3.7, -1.8, 0.9, -0.2, 0.0], dtype=np.float64)
        static_result = static_pipeline.compress(vector)
        dynamic_result = dynamic_pipeline.compress(vector)

        self.assertTrue(np.array_equal(static_pipeline._levels, dynamic_pipeline._levels))
        self.assertTrue(np.array_equal(static_result.vector, dynamic_result.vector))
        self.assertEqual(static_result.stats, dynamic_result.stats)

    def test_quantized_output_alphabet_has_zero_plus_signed_levels(self) -> None:
        for bits in (1, 3, 4):
            with self.subTest(bits=bits):
                config = RunConfig(
                    dataset="a9a",
                    method="ef21",
                    compressor_pipeline="db",
                    quantizer_family="db",
                    dynamic_mode="static",
                    bits_per_value=bits,
                    p=2,
                )
                pipeline = build_pipeline(config=config, dimension=8)
                pipeline.prepare_round([np.array([8.0])])
                levels = pipeline._levels.copy()
                probe = np.concatenate((np.array([0.0]), levels, -levels))
                result = pipeline.compress(probe)
                observed_states = {float(value) for value in result.vector}

                self.assertEqual(len(levels), 2 ** (bits - 1))
                self.assertEqual(len(observed_states), 1 + 2 * len(levels))
                self.assertEqual(len(observed_states), 2**bits + 1)
                self.assertIn(0.0, observed_states)

    def test_bits_for_values_contract_can_hold_even_when_alphabet_has_more_than_two_to_b_states(
        self,
    ) -> None:
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="db",
            quantizer_family="db",
            dynamic_mode="static",
            bits_per_value=4,
            p=2,
        )
        pipeline = build_pipeline(config=config, dimension=17)
        pipeline.prepare_round([np.array([8.0])])
        levels = pipeline._levels.copy()
        probe = np.concatenate((np.array([0.0]), levels, -levels))
        result = pipeline.compress(probe)
        observed_states = {float(value) for value in result.vector}

        self.assertEqual(result.stats.transmitted_coordinates, probe.size)
        self.assertEqual(result.stats.bits_for_values, probe.size * config.bits_per_value)
        self.assertEqual(result.stats.bits_for_indices, 0)
        self.assertGreater(len(observed_states), 2**config.bits_per_value)


if __name__ == "__main__":
    unittest.main()
