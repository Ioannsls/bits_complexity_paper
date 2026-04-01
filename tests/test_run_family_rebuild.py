from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.common.config import RunConfig
from bits_complexity.experiments import run_family
from bits_complexity.experiments.presets import family_db_configs, family_manifest_path


def _write_metrics_csv(path: Path, records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kbits_per_n", "grad_norm_sq"])
        writer.writeheader()
        for idx in range(records):
            writer.writerow({"kbits_per_n": idx * 0.1, "grad_norm_sq": 1.0 / (idx + 1)})


def _write_existing_run(
    output_dir: Path,
    config: RunConfig,
    *,
    run_name: str,
    records: int,
) -> tuple[Path, Path]:
    run_dir = output_dir / "runs" / config.dataset / run_name
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    _write_metrics_csv(metrics_path, records=records)
    payload = {
        **config.to_dict(),
        "resolved_quantizer_family": config.resolved_quantizer_family,
        "records": records,
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return metrics_path, config_path


class RunFamilyRebuildTest(unittest.TestCase):
    def test_rebuild_from_existing_uses_existing_runs_and_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            dataset = "a9a"
            db_configs = family_db_configs(dataset, output_dir)
            selected_label = r"$\mathcal{D}_B$, FP4"
            selected_path_fragment = "run_best"
            for idx, (_label, config) in enumerate(db_configs):
                _write_existing_run(
                    output_dir,
                    config,
                    run_name=f"run_{idx}",
                    records=10 + idx,
                )
            b4_config = next(config for label, config in db_configs if label == selected_label)
            _write_existing_run(output_dir, b4_config, run_name="run_best", records=999)

            argv = [
                "run_family",
                "--dataset",
                dataset,
                "--family",
                "db",
                "--output-dir",
                str(output_dir),
                "--rebuild-from-existing",
            ]
            with patch.object(sys, "argv", argv):
                with patch("bits_complexity.experiments.run_family.run_experiment") as run_mock:
                    with patch(
                        "bits_complexity.experiments.run_family.plot_family_from_manifest"
                    ) as plot_mock:
                        run_family.main()
            run_mock.assert_not_called()
            self.assertEqual(plot_mock.call_count, 2)
            self.assertEqual(plot_mock.call_args_list[0].kwargs["y_key"], "grad_norm_sq")
            self.assertEqual(plot_mock.call_args_list[1].kwargs["y_key"], "objective")

            manifest_path = family_manifest_path(output_dir, dataset, "db")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["entries"]), 5)
            self.assertIn("plot_paths", payload)
            self.assertIn("grad_norm_sq", payload["plot_paths"])
            self.assertIn("objective", payload["plot_paths"])
            selected_entry = next(
                entry for entry in payload["entries"] if entry["label"] == selected_label
            )
            self.assertIn(selected_path_fragment, selected_entry["csv_path"])

    def test_rebuild_from_existing_fails_when_required_curves_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            dataset = "a9a"
            db_configs = family_db_configs(dataset, output_dir)
            _write_existing_run(output_dir, db_configs[0][1], run_name="run_only", records=10)

            argv = [
                "run_family",
                "--dataset",
                dataset,
                "--family",
                "db",
                "--output-dir",
                str(output_dir),
                "--rebuild-from-existing",
            ]
            with patch.object(sys, "argv", argv):
                with patch("bits_complexity.experiments.run_family.run_experiment") as run_mock:
                    with self.assertRaisesRegex(ValueError, "Missing existing runs"):
                        run_family.main()
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
