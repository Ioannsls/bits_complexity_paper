from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.common.io import read_csv, write_csv, write_json


class IoHelpersTest(unittest.TestCase):
    def test_write_json_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "config.json"
            write_json(path, {"a": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["a"], 1)

    def test_write_and_read_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.csv"
            rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            write_csv(path, ["x", "y"], rows)
            loaded = read_csv(path)
            self.assertEqual(loaded[0]["x"], "1")
            self.assertEqual(loaded[1]["y"], "4")


if __name__ == "__main__":
    unittest.main()
