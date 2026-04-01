from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.common.config import RunConfig
from bits_complexity.experiments.presets import (
    family_db_configs,
    family_db_topk_configs,
    family_du_configs,
    family_du_topk_configs,
    family_manifest_path,
    family_plot_path,
    family_plot_x_max,
    family_static_dynamic_b3_configs,
    family_static_dynamic_configs,
    family_static_dynamic_du_b3_configs,
    family_static_dynamic_du_configs,
    publication_family_title,
)
from bits_complexity.experiments.runner import (
    DEFAULT_LR_SEARCH_ITERATIONS,
    LR_SEARCH_SAFETY_SHRINK,
    _is_monotone_trial,
    resolve_learning_rate,
    run_experiment,
)
from bits_complexity.plots.builder import plot_family_from_manifest
from bits_complexity.problems.logistic import LogisticProblem


def _write_binary_libsvm(path: Path, rows: list[tuple[float, dict[int, float]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label, features in rows:
            serialized = " ".join(f"{index}:{value}" for index, value in sorted(features.items()))
            handle.write(f"{label} {serialized}\n")


class RunnerTest(unittest.TestCase):
    def test_logistic_smoothness_constant_grows_with_l2(self) -> None:
        features = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]], dtype=np.float64)
        labels = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        base = LogisticProblem(
            train_features=features,
            train_labels=labels,
            test_features=features,
            test_labels=labels,
            l2_reg=0.0,
        )
        regularized = LogisticProblem(
            train_features=features,
            train_labels=labels,
            test_features=features,
            test_labels=labels,
            l2_reg=0.5,
        )
        self.assertGreater(base.smoothness_constant(), 0.0)
        self.assertGreater(regularized.smoothness_constant(), base.smoothness_constant())

    def test_resolve_learning_rate_enforces_monotone_prefix(self) -> None:
        features = np.array(
            [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.0], [-0.5, -1.0]],
            dtype=np.float64,
        )
        labels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        problem = LogisticProblem(
            train_features=features,
            train_labels=labels,
            test_features=features,
            test_labels=labels,
            l2_reg=0.1,
        )
        client_data = [(features[:2], labels[:2]), (features[2:], labels[2:])]
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="fp32",
            quantizer_family="none",
            clients=2,
            learning_rate=10.0,
            learning_rate_auto=True,
            max_iterations=40,
        )
        resolution = resolve_learning_rate(config=config, problem=problem, client_data=client_data)
        self.assertLess(resolution.resolved_learning_rate, config.learning_rate)
        manual_low = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="fp32",
            quantizer_family="none",
            clients=2,
            learning_rate=resolution.resolved_learning_rate,
        )
        self.assertTrue(
            _is_monotone_trial(
                config=manual_low,
                problem=problem,
                client_data=client_data,
                trial_iterations=DEFAULT_LR_SEARCH_ITERATIONS,
            )
        )
        failing_learning_rate = resolution.resolved_learning_rate / LR_SEARCH_SAFETY_SHRINK
        while _is_monotone_trial(
            config=RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                clients=2,
                learning_rate=failing_learning_rate,
            ),
            problem=problem,
            client_data=client_data,
            trial_iterations=DEFAULT_LR_SEARCH_ITERATIONS,
        ):
            failing_learning_rate *= 2.0
        self.assertFalse(
            _is_monotone_trial(
                config=RunConfig(
                    dataset="a9a",
                    method="ef21",
                    compressor_pipeline="fp32",
                    quantizer_family="none",
                    clients=2,
                    learning_rate=failing_learning_rate,
                ),
                problem=problem,
                client_data=client_data,
                trial_iterations=DEFAULT_LR_SEARCH_ITERATIONS,
            )
        )

    def test_dry_run_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            _write_binary_libsvm(
                datasets_dir / "a9a.txt",
                [
                    (1.0, {1: 1.0, 2: 0.2}),
                    (-1.0, {1: -1.0, 2: 0.3}),
                    (1.0, {1: 0.8, 2: -0.1}),
                    (-1.0, {1: -0.7, 2: -0.4}),
                ],
            )
            _write_binary_libsvm(
                datasets_dir / "a9a_test.txt",
                [
                    (1.0, {1: 1.0, 2: 0.1}),
                    (-1.0, {1: -0.9, 2: 0.2}),
                ],
            )
            config = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="db_topk",
                dynamic_mode="static",
                bits_per_value=4,
                dry_run=True,
                output_dir=root / "outputs",
            )
            artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
            self.assertTrue(artifacts.csv_path.exists())
            self.assertTrue(artifacts.config_path.exists())
            with artifacts.csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_short_run_logs_monotone_communication_and_respects_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            train_rows = []
            for idx in range(20):
                label = 1.0 if idx % 2 == 0 else -1.0
                train_rows.append((label, {1: label, 2: idx / 20.0}))
            test_rows = [(1.0, {1: 1.0, 2: 0.1}), (-1.0, {1: -1.0, 2: 0.2})]
            _write_binary_libsvm(datasets_dir / "a9a.txt", train_rows)
            _write_binary_libsvm(datasets_dir / "a9a_test.txt", test_rows)
            config = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="topk",
                dynamic_mode="disabled",
                output_dir=root / "outputs",
                cutoff_kbits_per_n=0.05,
                max_iterations=10,
            )
            artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
            with artifacts.csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]["iteration"], "0")
            self.assertEqual(rows[0]["cum_bits"], "0")
            self.assertEqual(rows[0]["kbits_per_n"], "0.0")
            self.assertEqual(rows[0]["step_bits"], "0")
            cum_bits = [int(row["cum_bits"]) for row in rows]
            self.assertEqual(cum_bits, sorted(cum_bits))
            self.assertGreaterEqual(float(rows[-1]["kbits_per_n"]), config.cutoff_kbits_per_n)
            self.assertIn("grad_norm_sq", rows[0])
            with artifacts.config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["records"], len(rows))

    def test_auto_learning_rate_is_saved_and_prefix_is_monotone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            train_rows = []
            for idx in range(64):
                label = 1.0 if idx % 2 == 0 else -1.0
                train_rows.append((label, {1: label, 2: (idx % 5) / 5.0}))
            test_rows = [(1.0, {1: 1.0, 2: 0.1}), (-1.0, {1: -1.0, 2: 0.2})]
            _write_binary_libsvm(datasets_dir / "a9a.txt", train_rows)
            _write_binary_libsvm(datasets_dir / "a9a_test.txt", test_rows)
            config = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                clients=2,
                output_dir=root / "outputs",
                max_iterations=160,
                cutoff_kbits_per_n=100.0,
                learning_rate=10.0,
                learning_rate_auto=True,
            )
            artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
            with artifacts.csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            prefix = [
                float(row["grad_norm_sq"]) for row in rows[: DEFAULT_LR_SEARCH_ITERATIONS + 1]
            ]
            self.assertGreaterEqual(len(prefix), DEFAULT_LR_SEARCH_ITERATIONS + 1)
            for previous_value, current_value in zip(prefix, prefix[1:]):
                tolerance = 1e-12 * max(1.0, previous_value)
                self.assertLessEqual(current_value, previous_value + tolerance)
            with artifacts.config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["learning_rate_auto"])
            self.assertEqual(payload["learning_rate_multiplier"], 1.0)
            self.assertEqual(
                payload["base_resolved_learning_rate"], payload["resolved_learning_rate"]
            )
            self.assertGreater(payload["resolved_learning_rate"], 0.0)
            self.assertEqual(
                payload["learning_rate_search_iterations"], DEFAULT_LR_SEARCH_ITERATIONS
            )
            self.assertGreater(payload["learning_rate_search_trial_runs"], 0)

    def test_manual_learning_rate_multiplier_scales_effective_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            _write_binary_libsvm(
                datasets_dir / "a9a.txt",
                [(1.0, {1: 1.0}), (-1.0, {1: -1.0}), (1.0, {1: 0.8}), (-1.0, {1: -0.8})],
            )
            _write_binary_libsvm(
                datasets_dir / "a9a_test.txt", [(1.0, {1: 1.0}), (-1.0, {1: -1.0})]
            )
            config = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                output_dir=root / "outputs",
                max_iterations=2,
                cutoff_kbits_per_n=100.0,
                learning_rate=0.25,
                learning_rate_multiplier=2.0,
            )
            artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
            with artifacts.config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertFalse(payload["learning_rate_auto"])
            self.assertEqual(payload["learning_rate_multiplier"], 2.0)
            self.assertEqual(payload["base_resolved_learning_rate"], 0.25)
            self.assertEqual(payload["resolved_learning_rate"], 0.5)

    def test_auto_learning_rate_multiplier_scales_final_resolved_learning_rate(self) -> None:
        features = np.array(
            [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.0], [-0.5, -1.0]],
            dtype=np.float64,
        )
        labels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        problem = LogisticProblem(
            train_features=features,
            train_labels=labels,
            test_features=features,
            test_labels=labels,
            l2_reg=0.1,
        )
        client_data = [(features[:2], labels[:2]), (features[2:], labels[2:])]
        config = RunConfig(
            dataset="a9a",
            method="ef21",
            compressor_pipeline="fp32",
            quantizer_family="none",
            clients=2,
            learning_rate=10.0,
            learning_rate_auto=True,
            learning_rate_multiplier=2.0,
            max_iterations=40,
        )
        resolution = resolve_learning_rate(config=config, problem=problem, client_data=client_data)
        self.assertEqual(resolution.multiplier, 2.0)
        self.assertAlmostEqual(
            resolution.resolved_learning_rate,
            resolution.base_resolved_learning_rate * 2.0,
        )

    def test_reuse_existing_csv_on_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            _write_binary_libsvm(
                datasets_dir / "a9a.txt",
                [(1.0, {1: 1.0}), (-1.0, {1: -1.0}), (1.0, {1: 0.8}), (-1.0, {1: -0.8})],
            )
            _write_binary_libsvm(
                datasets_dir / "a9a_test.txt", [(1.0, {1: 1.0}), (-1.0, {1: -1.0})]
            )
            config = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="topk",
                quantizer_family="none",
                output_dir=root / "outputs",
                max_iterations=2,
                cutoff_kbits_per_n=0.01,
            )
            first = run_experiment(config=config, datasets_dir=datasets_dir)
            second = run_experiment(config=config, datasets_dir=datasets_dir)
            self.assertFalse(first.reused_existing)
            self.assertTrue(second.reused_existing)

    def test_manual_and_auto_learning_rate_do_not_share_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            _write_binary_libsvm(
                datasets_dir / "a9a.txt",
                [(1.0, {1: 1.0}), (-1.0, {1: -1.0}), (1.0, {1: 0.8}), (-1.0, {1: -0.8})],
            )
            _write_binary_libsvm(
                datasets_dir / "a9a_test.txt", [(1.0, {1: 1.0}), (-1.0, {1: -1.0})]
            )
            manual = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                output_dir=root / "outputs",
                max_iterations=3,
                cutoff_kbits_per_n=100.0,
                learning_rate=0.01,
            )
            auto = RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                output_dir=root / "outputs",
                max_iterations=3,
                cutoff_kbits_per_n=100.0,
                learning_rate=0.01,
                learning_rate_auto=True,
            )
            manual_artifacts = run_experiment(config=manual, datasets_dir=datasets_dir)
            auto_artifacts = run_experiment(config=auto, datasets_dir=datasets_dir)
            self.assertNotEqual(manual_artifacts.csv_path, auto_artifacts.csv_path)
            self.assertNotEqual(manual_artifacts.config_path, auto_artifacts.config_path)

    def test_diana_run_logs_du_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir()
            train_rows = []
            for idx in range(20):
                label = 1.0 if idx % 2 == 0 else -1.0
                train_rows.append((label, {1: label, 2: idx / 20.0}))
            test_rows = [(1.0, {1: 1.0, 2: 0.1}), (-1.0, {1: -1.0, 2: 0.2})]
            _write_binary_libsvm(datasets_dir / "a9a.txt", train_rows)
            _write_binary_libsvm(datasets_dir / "a9a_test.txt", test_rows)
            config = RunConfig(
                dataset="a9a",
                method="diana",
                compressor_pipeline="db",
                quantizer_family="du",
                dynamic_mode="static",
                output_dir=root / "outputs",
                cutoff_kbits_per_n=0.05,
                max_iterations=2,
            )
            artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
            with artifacts.csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]["iteration"], "0")
            self.assertEqual(rows[0]["cum_bits"], "0")
            self.assertEqual(rows[1]["iteration"], "1")
            self.assertTrue(all(row["quantizer_family"] == "du" for row in rows))

    def test_publication_presets_cover_required_curves(self) -> None:
        output_dir = Path("outputs")
        self.assertEqual(len(family_db_configs("a9a", output_dir)), 5)
        self.assertEqual(len(family_du_configs("a9a", output_dir)), 5)
        self.assertEqual(len(family_db_topk_configs("a9a", output_dir)), 5)
        self.assertEqual(len(family_du_topk_configs("a9a", output_dir)), 5)
        self.assertEqual(len(family_static_dynamic_configs("a9a", output_dir)), 6)
        self.assertEqual(len(family_static_dynamic_b3_configs("a9a", output_dir)), 6)
        self.assertEqual(len(family_static_dynamic_du_configs("a9a", output_dir)), 6)
        self.assertEqual(len(family_static_dynamic_du_b3_configs("a9a", output_dir)), 6)

    def test_publication_family_titles_follow_canonical_format(self) -> None:
        self.assertEqual(
            publication_family_title("db"),
            r"EF21 with $\mathcal{D}_B$ compressor",
        )
        self.assertEqual(
            publication_family_title("du"),
            r"EF21 with $\mathcal{D}_U$ compressor",
        )
        self.assertEqual(
            publication_family_title("db_topk"),
            r"EF21 with $\mathcal{D}_B$ + Top10% compressors",
        )
        self.assertEqual(
            publication_family_title("du_topk"),
            r"EF21 with $\mathcal{D}_U$ + Top10% compressors",
        )
        self.assertEqual(
            publication_family_title("static_dynamic"),
            r"Static vs Dynamic with $\mathcal{D}_B$ compressor, FP4",
        )
        self.assertEqual(
            publication_family_title("static_dynamic_b3"),
            r"Static vs Dynamic with $\mathcal{D}_B$ compressor, FP3",
        )
        self.assertEqual(
            publication_family_title("static_dynamic_du"),
            r"Static vs Dynamic with $\mathcal{D}_U$ compressor, FP4",
        )
        self.assertEqual(
            publication_family_title("static_dynamic_du_b3"),
            r"Static vs Dynamic with $\mathcal{D}_U$ compressor, FP3",
        )

    def test_db_publication_family_uses_reference_plus_3_to_6_bits(self) -> None:
        output_dir = Path("outputs")
        family = family_db_configs("a9a", output_dir)
        labels = [label for label, _ in family]
        self.assertEqual(
            labels,
            [
                "FP32",
                r"$\mathcal{D}_B$, FP3",
                r"$\mathcal{D}_B$, FP4",
                r"$\mathcal{D}_B$, FP5",
                r"$\mathcal{D}_B$, FP6",
            ],
        )
        reference = family[0][1]
        self.assertEqual(reference.compressor_pipeline, "fp32")
        self.assertEqual(reference.bits_per_value, 32)

        for _, config in family[1:]:
            self.assertEqual(config.dataset, "a9a")
            self.assertEqual(config.method, "ef21")
            self.assertEqual(config.compressor_pipeline, "db")
            self.assertEqual(config.quantizer_family, "db")
            self.assertEqual(config.dynamic_mode, "dynamic")
            self.assertIn(config.bits_per_value, {3, 4, 5, 6})

    def test_du_topk_publication_family_uses_topk_reference_plus_3_to_6_bits(self) -> None:
        output_dir = Path("outputs")
        family = family_du_topk_configs("a9a", output_dir)
        labels = [label for label, _ in family]
        self.assertEqual(
            labels,
            [
                "FP32, topk=10%",
                r"$\mathcal{D}_U$, FP3, topk=10%",
                r"$\mathcal{D}_U$, FP4, topk=10%",
                r"$\mathcal{D}_U$, FP5, topk=10%",
                r"$\mathcal{D}_U$, FP6, topk=10%",
            ],
        )
        reference = family[0][1]
        self.assertEqual(reference.compressor_pipeline, "topk")
        self.assertEqual(reference.bits_per_value, 32)

        for _, config in family[1:]:
            self.assertEqual(config.dataset, "a9a")
            self.assertEqual(config.method, "ef21")
            self.assertEqual(config.compressor_pipeline, "db_topk")
            self.assertEqual(config.quantizer_family, "du")
            self.assertEqual(config.dynamic_mode, "dynamic")
            self.assertIn(config.bits_per_value, {3, 4, 5, 6})

    def test_static_dynamic_presets_cover_p_and_mode_combinations_for_b4(self) -> None:
        output_dir = Path("outputs")
        family = family_static_dynamic_configs("a9a", output_dir, cutoff_kbits_per_n=200.0)
        grouped: dict[int, list[tuple[str, RunConfig]]] = {}
        for label, config in family:
            grouped.setdefault(config.p, []).append((label, config))
            self.assertEqual(config.dataset, "a9a")
            self.assertEqual(config.method, "ef21")
            self.assertEqual(config.compressor_pipeline, "db")
            self.assertEqual(config.quantizer_family, "db")
            self.assertEqual(config.output_dir, output_dir)
            self.assertEqual(config.bits_per_value, 4)
            self.assertEqual(config.cutoff_kbits_per_n, 200.0)

        self.assertEqual(set(grouped), {2, 4, 8})
        for p_value, configs in grouped.items():
            self.assertEqual(len(configs), 2)
            self.assertEqual(
                {config.dynamic_mode for _, config in configs},
                {"static", "dynamic"},
            )
            self.assertEqual(
                {label for label, _ in configs},
                {
                    rf"$\mathcal{{D}}_B$, dynamic, p={p_value}, FP4",
                    rf"$\mathcal{{D}}_B$, static, p={p_value}, FP4",
                },
            )

    def test_static_dynamic_b3_presets_scale_cutoff(self) -> None:
        output_dir = Path("outputs")
        family = family_static_dynamic_b3_configs("w8a", output_dir, cutoff_kbits_per_n=70.0)
        self.assertEqual(len(family), 6)
        for label, config in family:
            self.assertEqual(config.quantizer_family, "db")
            self.assertEqual(config.bits_per_value, 3)
            self.assertEqual(label, rf"$\mathcal{{D}}_B$, {config.dynamic_mode}, p={config.p}, FP3")
            self.assertEqual(config.cutoff_kbits_per_n, 52.5)

    def test_static_dynamic_du_presets_use_du_labels_and_scaled_cutoff(self) -> None:
        output_dir = Path("outputs")
        family = family_static_dynamic_du_configs("w8a", output_dir, cutoff_kbits_per_n=70.0)
        self.assertEqual(len(family), 6)
        for label, config in family:
            self.assertIn(config.dynamic_mode, label)
            self.assertEqual(
                label,
                rf"$\mathcal{{D}}_U$, {config.dynamic_mode}, p={config.p}, FP{config.bits_per_value}",
            )
            self.assertEqual(config.quantizer_family, "du")
            self.assertEqual(config.compressor_pipeline, "db")
            self.assertEqual(config.bits_per_value, 4)
            self.assertEqual(config.cutoff_kbits_per_n, 70.0)

    def test_static_dynamic_du_b3_presets_scale_cutoff(self) -> None:
        output_dir = Path("outputs")
        family = family_static_dynamic_du_b3_configs("w8a", output_dir, cutoff_kbits_per_n=70.0)
        self.assertEqual(len(family), 6)
        for label, config in family:
            self.assertEqual(config.quantizer_family, "du")
            self.assertEqual(config.bits_per_value, 3)
            self.assertEqual(label, rf"$\mathcal{{D}}_U$, {config.dynamic_mode}, p={config.p}, FP3")
            self.assertEqual(config.cutoff_kbits_per_n, 52.5)

    def test_family_plot_paths_follow_new_contract(self) -> None:
        output_dir = Path("outputs")
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "db"),
            output_dir / "plots" / "1_family_grad_norm" / "ef21_db_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "db", y_key="objective"),
            output_dir / "plots" / "2_family_objective" / "ef21_db_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "du_topk"),
            output_dir / "plots" / "1_family_grad_norm" / "ef21_du_topk_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "static_dynamic"),
            output_dir
            / "plots"
            / "3_dynamic_vs_static_grad_norm"
            / "ef21_static_vs_dynamic_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "static_dynamic_b3"),
            output_dir
            / "plots"
            / "3_dynamic_vs_static_grad_norm"
            / "ef21_static_vs_dynamic_b3_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "static_dynamic_du"),
            output_dir
            / "plots"
            / "3_dynamic_vs_static_grad_norm"
            / "ef21_static_vs_dynamic_du_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "static_dynamic_du_b3"),
            output_dir
            / "plots"
            / "3_dynamic_vs_static_grad_norm"
            / "ef21_static_vs_dynamic_du_b3_a9a.png",
        )
        self.assertEqual(
            family_plot_path(output_dir, "a9a", "static_dynamic_du", y_key="objective"),
            output_dir
            / "plots"
            / "4_dynamic_vs_static_objective"
            / "ef21_static_vs_dynamic_du_a9a.png",
        )

    def test_family_plot_x_max_overrides_for_w8a(self) -> None:
        self.assertEqual(family_plot_x_max("w8a", "db_topk"), 3.0)
        self.assertEqual(family_plot_x_max("w8a", "du_topk"), 3.0)
        self.assertEqual(family_plot_x_max("w8a", "db"), 33.0)
        self.assertEqual(family_plot_x_max("w8a", "du"), 33.0)
        self.assertEqual(family_plot_x_max("w8a", "static_dynamic"), 40.0)
        self.assertEqual(family_plot_x_max("w8a", "static_dynamic_du"), 40.0)
        self.assertIsNone(family_plot_x_max("w8a", "static_dynamic_b3"))
        self.assertIsNone(family_plot_x_max("a9a", "db"))

    def test_plot_manifest_contract_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plot_path = root / "plots" / "ef21_db_a9a.png"
            csv_a = root / "curve_a.csv"
            csv_b = root / "curve_b.csv"
            for csv_path, values in (
                (csv_a, [(0.0, 1.5), (0.1, 1.0), (0.3, 0.7)]),
                (csv_b, [(0.0, 1.6), (0.1, 1.2), (0.2, 0.8)]),
            ):
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                    writer.writeheader()
                    for x_val, y_val in values:
                        writer.writerow({"kbits_per_n": x_val, "grad_norm_sq": y_val})
            manifest_path = family_manifest_path(root, "a9a", "db")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "a9a",
                        "family": "db",
                        "plot_path": str(plot_path),
                        "entries": [
                            {"label": "curve-a", "csv_path": str(csv_a)},
                            {"label": "curve-b", "csv_path": str(csv_b)},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            try:
                output_path = plot_family_from_manifest(manifest_path)
                self.assertEqual(output_path, plot_path)
            except RuntimeError as exc:
                self.assertIn("matplotlib is required", str(exc))


if __name__ == "__main__":
    unittest.main()
