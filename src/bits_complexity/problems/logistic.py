from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LogisticProblem:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    l2_reg: float = 0.0

    @property
    def dimension(self) -> int:
        return int(self.train_features.shape[1])

    def initial_point(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=np.float64)

    @staticmethod
    def _safe_linear_response(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = features @ weights
        return np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)

    def objective(
        self,
        weights: np.ndarray,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        x = self.train_features if features is None else features
        y = self.train_labels if labels is None else labels
        margins = y * self._safe_linear_response(x, weights)
        loss = np.logaddexp(0.0, -margins).mean()
        reg = 0.5 * self.l2_reg * float(np.dot(weights, weights))
        return float(loss + reg)

    def gradient(self, weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros_like(weights)
        margins = labels * self._safe_linear_response(features, weights)
        coeff = -labels / (1.0 + np.exp(margins))
        grad = (coeff[:, None] * features).mean(axis=0)
        if self.l2_reg:
            grad = grad + self.l2_reg * weights
        return grad

    def full_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.gradient(weights, self.train_features, self.train_labels)

    def smoothness_constant(self) -> float:
        sample_count = max(1, int(self.train_features.shape[0]))
        gram = (self.train_features.T @ self.train_features) / float(sample_count)
        max_eigenvalue = float(np.linalg.eigvalsh(gram)[-1]) if gram.size else 0.0
        return 0.25 * max(max_eigenvalue, 0.0) + self.l2_reg

    def grad_norm_sq(self, weights: np.ndarray) -> float:
        grad = self.full_gradient(weights)
        return float(np.dot(grad, grad))

    def accuracy(self, weights: np.ndarray) -> float:
        logits = self._safe_linear_response(self.test_features, weights)
        predictions = np.where(logits >= 0.0, 1.0, -1.0)
        return float((predictions == self.test_labels).mean())
