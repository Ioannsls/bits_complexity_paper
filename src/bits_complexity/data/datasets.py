from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bits_complexity.data.libsvm import DenseDataset, load_libsvm_dense, scan_dimension


@dataclass
class DatasetBundle:
    name: str
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.train_features.shape[1])

    @property
    def train_size(self) -> int:
        return int(self.train_features.shape[0])


def _normalize_binary_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted({float(value) for value in labels})
    if unique == [-1.0, 1.0]:
        return labels.astype(np.float64)
    if unique == [0.0, 1.0]:
        return np.where(labels > 0, 1.0, -1.0)
    if unique == [1.0, 2.0]:
        return np.where(labels > 1.0, 1.0, -1.0)
    raise ValueError(f"Unsupported binary label encoding: {unique}")


def _split_train_test(
    dataset: DenseDataset,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = dataset.features
    labels = _normalize_binary_labels(dataset.labels)
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    test_size = max(1, int(round(features.shape[0] * test_ratio)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        features[train_indices],
        labels[train_indices],
        features[test_indices],
        labels[test_indices],
    )


def load_dataset(name: str, datasets_dir: Path, seed: int) -> DatasetBundle:
    if name == "mushrooms":
        raw = load_libsvm_dense(datasets_dir / "mushrooms.txt")
        train_x, train_y, test_x, test_y = _split_train_test(raw, test_ratio=0.2, seed=seed)
    elif name in {"a9a", "w8a"}:
        train_path = datasets_dir / f"{name}.txt"
        test_path = datasets_dir / f"{name}_test.txt"
        dimension = max(scan_dimension(train_path), scan_dimension(test_path))
        train = load_libsvm_dense(train_path, dimension=dimension)
        test = load_libsvm_dense(test_path, dimension=dimension)
        train_x, train_y = train.features, _normalize_binary_labels(train.labels)
        test_x, test_y = test.features, _normalize_binary_labels(test.labels)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DatasetBundle(
        name=name,
        train_features=train_x.astype(np.float64),
        train_labels=train_y.astype(np.float64),
        test_features=test_x.astype(np.float64),
        test_labels=test_y.astype(np.float64),
    )


def split_into_clients(
    features: np.ndarray,
    labels: np.ndarray,
    clients: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if clients <= 0:
        raise ValueError("clients must be positive")
    feature_splits = np.array_split(features, clients, axis=0)
    label_splits = np.array_split(labels, clients, axis=0)
    return list(zip(feature_splits, label_splits))
