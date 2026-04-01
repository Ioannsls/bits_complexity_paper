from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DenseDataset:
    features: np.ndarray
    labels: np.ndarray


def scan_dimension(path: Path) -> int:
    max_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            for item in line.split()[1:]:
                if ":" not in item:
                    continue
                index_str, _ = item.split(":", maxsplit=1)
                max_index = max(max_index, int(index_str))
    return max_index


def load_libsvm_dense(path: Path, dimension: Optional[int] = None) -> DenseDataset:
    if dimension is None:
        dimension = scan_dimension(path)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(float(parts[0]))
            row = np.zeros(dimension, dtype=np.float64)
            for item in parts[1:]:
                if ":" not in item:
                    continue
                index_str, value_str = item.split(":", maxsplit=1)
                row[int(index_str) - 1] = float(value_str)
            rows.append(row)
    return DenseDataset(
        features=np.vstack(rows) if rows else np.zeros((0, dimension), dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
    )
