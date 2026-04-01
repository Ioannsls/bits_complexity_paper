from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.data.datasets import load_dataset, split_into_clients


def _write_libsvm(path: Path, rows: list[tuple[float, dict[int, float]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label, features in rows:
            serialized = " ".join(f"{index}:{value}" for index, value in sorted(features.items()))
            handle.write(f"{label} {serialized}\n")


class DataLoadingSmokeTest(unittest.TestCase):
    def _prepare_datasets(self, datasets_dir: Path) -> None:
        _write_libsvm(
            datasets_dir / "mushrooms.txt",
            [
                (1.0, {1: 1.0, 2: 0.0}),
                (-1.0, {1: 0.0, 2: 1.0}),
                (1.0, {1: 1.0, 2: 1.0}),
                (-1.0, {1: 0.2, 2: 0.8}),
                (1.0, {1: 0.8, 2: 0.2}),
            ],
        )
        _write_libsvm(
            datasets_dir / "a9a.txt",
            [
                (1.0, {1: 1.0, 2: 0.0}),
                (-1.0, {1: 0.0, 2: 1.0}),
                (1.0, {1: 0.8, 2: 0.2}),
                (-1.0, {1: 0.2, 2: 0.8}),
            ],
        )
        _write_libsvm(
            datasets_dir / "a9a_test.txt",
            [
                (1.0, {1: 1.0, 2: 0.1}),
                (-1.0, {1: 0.1, 2: 1.0}),
            ],
        )
        _write_libsvm(
            datasets_dir / "w8a.txt",
            [
                (1.0, {1: 1.0, 3: 0.2}),
                (-1.0, {2: 1.0, 3: 0.1}),
                (1.0, {1: 0.9, 2: 0.1}),
                (-1.0, {2: 0.9, 1: 0.1}),
            ],
        )
        _write_libsvm(
            datasets_dir / "w8a_test.txt",
            [
                (1.0, {1: 1.0, 2: 0.2}),
                (-1.0, {2: 1.0, 1: 0.2}),
            ],
        )

    def test_load_all_required_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir)
            self._prepare_datasets(datasets_dir)
            for dataset_name in ("mushrooms", "a9a", "w8a"):
                bundle = load_dataset(dataset_name, datasets_dir=datasets_dir, seed=42)
                self.assertGreater(bundle.train_features.shape[0], 0)
                self.assertGreater(bundle.test_features.shape[0], 0)
                self.assertEqual(bundle.train_features.shape[1], bundle.test_features.shape[1])

    def test_client_split_has_exact_number_of_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir)
            self._prepare_datasets(datasets_dir)
            bundle = load_dataset("a9a", datasets_dir=datasets_dir, seed=42)
            client_data = split_into_clients(bundle.train_features, bundle.train_labels, clients=10)
            self.assertEqual(len(client_data), 10)
            total_rows = sum(features.shape[0] for features, _ in client_data)
            self.assertEqual(total_rows, bundle.train_features.shape[0])


if __name__ == "__main__":
    unittest.main()
