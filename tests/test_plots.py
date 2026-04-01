from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.plots.builder import (
    CurveData,
    _compact_curve_label,
    _marker_positions,
    _normalize_plot_title,
    build_family_plot,
    plot_family_from_csvs,
    plot_family_from_manifest,
)


class PlotBuilderTest(unittest.TestCase):
    def test_build_family_plot_default_x_label(self) -> None:
        self.assertEqual(build_family_plot.__defaults__[0], "#Kbits/n")

    def test_compact_curve_label_rules(self) -> None:
        self.assertEqual(_compact_curve_label("EF21 + fp32"), "FP32")
        self.assertEqual(_compact_curve_label("EF21 + fp32, topk=10%"), "FP32, topk=10%")
        self.assertEqual(
            _compact_curve_label("EF21 + biased dynamic dtype, b=4"),
            r"$\mathcal{D}_B$, FP4",
        )
        self.assertEqual(
            _compact_curve_label("EF21 + biased deynamic dtype, b=4"),
            r"$\mathcal{D}_B$, FP4",
        )
        self.assertEqual(
            _compact_curve_label("EF21 + unbiased dynamic dtype, b=6, topk=10%"),
            r"$\mathcal{D}_U$, FP6, topk=10%",
        )
        self.assertEqual(
            _compact_curve_label("EF21 + unbiased deynamic dtype, b=3"),
            r"$\mathcal{D}_U$, FP3",
        )
        self.assertEqual(_compact_curve_label("dynamic, p=8, b=3"), "dynamic, p=8, FP3")
        self.assertEqual(_compact_curve_label("dynamic, p=8, b3"), "dynamic, p=8, FP3")
        self.assertEqual(_compact_curve_label("dynamic, p=8, fp3"), "dynamic, p=8, FP3")
        self.assertEqual(_compact_curve_label("static, p=2, b=4"), "static, p=2, FP4")
        self.assertEqual(
            _compact_curve_label("EF21 + unbiased dynamic dtype, fp3, topk=10%"),
            r"$\mathcal{D}_U$, FP3, topk=10%",
        )
        self.assertEqual(_compact_curve_label("custom label"), "custom label")

    def test_compact_curve_label_handles_math_and_mode_from_presets(self) -> None:
        self.assertEqual(
            _compact_curve_label(r"$\mathcal{D}_U$, dynamic, p=4, FP3"),
            r"$\mathcal{D}_U$, dynamic, p=4, FP3",
        )

    def test_normalize_plot_title_includes_dataset_once(self) -> None:
        self.assertEqual(
            _normalize_plot_title("a9a", "EF21 + biased dynamic dtype"),
            "a9a: EF21 + biased dynamic dtype",
        )
        self.assertEqual(
            _normalize_plot_title("a9a", "a9a: EF21 + biased dynamic dtype"),
            "a9a: EF21 + biased dynamic dtype",
        )

    def test_marker_positions_are_fixed_count_and_point_based(self) -> None:
        self.assertEqual(_marker_positions(0, marker_count=10), [])
        self.assertEqual(_marker_positions(1, marker_count=10), [0] * 10)
        self.assertEqual(_marker_positions(3, marker_count=10), [0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(_marker_positions(10, marker_count=10), list(range(10)))
        self.assertEqual(
            _marker_positions(20, marker_count=10), [0, 2, 4, 6, 8, 11, 13, 15, 17, 19]
        )

    def test_plot_builder_requires_matplotlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "curve.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                writer.writeheader()
                writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
                writer.writerow({"kbits_per_n": 0.2, "grad_norm_sq": 0.8})
            try:
                plot_family_from_csvs(
                    curve_specs=[("curve", csv_path)],
                    output_path=root / "plot.png",
                    title="test",
                )
            except RuntimeError as exc:
                self.assertIn("matplotlib is required", str(exc))

    def test_common_cutoff_uses_shortest_curve_budget(self) -> None:
        curves = [
            CurveData(label="a", x=[0.0, 0.1, 0.3], y=[1.4, 1.0, 0.5]),
            CurveData(label="b", x=[0.0, 0.1, 0.2], y=[1.5, 1.2, 0.8]),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                build_family_plot(
                    curves=curves, output_path=Path(tmp_dir) / "plot.png", title="family"
                )
            except RuntimeError as exc:
                self.assertIn("matplotlib is required", str(exc))

    def test_manifest_title_uses_canonical_family_title_and_forwards_xmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "curve.csv"
            manifest_path = root / "manifest.json"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                writer.writeheader()
                writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "a9a",
                        "family": "db",
                        "x_max": 30,
                        "title": "Static vs dynamic, biased dynamic dtype",
                        "plot_path": str(root / "plot.png"),
                        "entries": [{"label": "curve", "csv_path": str(csv_path)}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("bits_complexity.plots.builder.plot_family_from_csvs") as plot_mock:
                output_path = plot_family_from_manifest(manifest_path)
                self.assertEqual(output_path, root / "plot.png")
                plot_mock.assert_called_once()
                self.assertEqual(
                    plot_mock.call_args.kwargs["title"],
                    r"a9a: EF21 with $\mathcal{D}_B$ compressor",
                )
                self.assertEqual(plot_mock.call_args.kwargs["x_max"], 30.0)

    def test_manifest_title_falls_back_to_payload_for_unknown_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "curve.csv"
            manifest_path = root / "manifest.json"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                writer.writeheader()
                writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "a9a",
                        "family": "custom_family",
                        "title": "Custom family title",
                        "plot_path": str(root / "plot.png"),
                        "entries": [{"label": "curve", "csv_path": str(csv_path)}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("bits_complexity.plots.builder.plot_family_from_csvs") as plot_mock:
                output_path = plot_family_from_manifest(manifest_path)
                self.assertEqual(output_path, root / "plot.png")
                self.assertEqual(plot_mock.call_args.kwargs["title"], "a9a: Custom family title")

    def test_manifest_plot_paths_selects_output_by_y_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "curve.csv"
            manifest_path = root / "manifest.json"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["kbits_per_n", "grad_norm_sq", "objective"]
                )
                writer.writeheader()
                writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0, "objective": 0.5})
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "a9a",
                        "family": "db",
                        "plot_path": str(root / "grad.png"),
                        "plot_paths": {
                            "grad_norm_sq": str(root / "grad.png"),
                            "objective": str(root / "objective.png"),
                        },
                        "entries": [{"label": "curve", "csv_path": str(csv_path)}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("bits_complexity.plots.builder.plot_family_from_csvs") as plot_mock:
                output_path = plot_family_from_manifest(manifest_path, y_key="objective")
                self.assertEqual(output_path, root / "objective.png")
                self.assertEqual(plot_mock.call_args.kwargs["y_key"], "objective")

    def test_objective_label_is_latex_f_of_x(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "curve.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "objective"])
                writer.writeheader()
                writer.writerow({"kbits_per_n": 0.1, "objective": 0.5})
            with mock.patch("bits_complexity.plots.builder.build_family_plot") as build_plot:
                plot_family_from_csvs(
                    curve_specs=[("curve", csv_path)],
                    output_path=root / "plot.png",
                    title="test",
                    y_key="objective",
                )
                self.assertEqual(build_plot.call_args.kwargs["y_label"], r"$f(x)$")
                self.assertFalse(build_plot.call_args.kwargs["log_x"])
                self.assertTrue(build_plot.call_args.kwargs["log_y"])

    def test_shared_legend_parts_move_to_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_a = root / "curve_a.csv"
            csv_b = root / "curve_b.csv"
            for csv_path in (csv_a, csv_b):
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                    writer.writeheader()
                    writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
            with mock.patch("bits_complexity.plots.builder.build_family_plot") as build_plot:
                plot_family_from_csvs(
                    curve_specs=[
                        (r"$\mathcal{D}_U$, dynamic, p=2, FP3", csv_a),
                        (r"$\mathcal{D}_U$, static, p=2, FP3", csv_b),
                    ],
                    output_path=root / "plot.png",
                    title="a9a - Static vs dynamic",
                )
                curves = build_plot.call_args.kwargs["curves"]
                self.assertEqual(curves[0].label, "dynamic")
                self.assertEqual(curves[1].label, "static")
                self.assertEqual(
                    build_plot.call_args.kwargs["title"],
                    r"a9a - Static vs dynamic, $\mathcal{D}_U$, p=2, FP3",
                )

    def test_shared_legend_parts_are_not_duplicated_in_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_a = root / "curve_a.csv"
            csv_b = root / "curve_b.csv"
            for csv_path in (csv_a, csv_b):
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                    writer.writeheader()
                    writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
            with mock.patch("bits_complexity.plots.builder.build_family_plot") as build_plot:
                plot_family_from_csvs(
                    curve_specs=[
                        (r"$\mathcal{D}_U$, dynamic, FP3", csv_a),
                        (r"$\mathcal{D}_U$, static, FP3", csv_b),
                    ],
                    output_path=root / "plot.png",
                    title=r"a9a - Static vs dynamic, $\mathcal{D}_U$, FP3",
                )
                self.assertEqual(
                    build_plot.call_args.kwargs["title"],
                    r"a9a - Static vs dynamic, $\mathcal{D}_U$, FP3",
                )

    def test_shared_topk_part_is_not_duplicated_when_title_has_top_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_a = root / "curve_a.csv"
            csv_b = root / "curve_b.csv"
            for csv_path in (csv_a, csv_b):
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
                    writer.writeheader()
                    writer.writerow({"kbits_per_n": 0.1, "grad_norm_sq": 1.0})
            with mock.patch("bits_complexity.plots.builder.build_family_plot") as build_plot:
                plot_family_from_csvs(
                    curve_specs=[
                        ("FP32, topk=10%", csv_a),
                        (r"$\mathcal{D}_U$, FP3, topk=10%", csv_b),
                    ],
                    output_path=root / "plot.png",
                    title=r"a9a: EF21 with $\mathcal{D}_U$ + Top10% compressors",
                )
                self.assertEqual(
                    build_plot.call_args.kwargs["title"],
                    r"a9a: EF21 with $\mathcal{D}_U$ + Top10% compressors",
                )


if __name__ == "__main__":
    unittest.main()
