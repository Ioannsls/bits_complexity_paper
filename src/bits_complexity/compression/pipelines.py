from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.base import CompressionResult, CompressionStats, index_bit_width


@dataclass
class PipelineState:
    value_bits: int = 32
    sparse: bool = False
    dynamic_rule: str = "disabled"
    quantizer_family: str = "none"


class CompressionPipeline:
    def __init__(self, config: RunConfig, dimension: int) -> None:
        self.config = config
        self.dimension = dimension
        self._current_amax: float | None = None
        self._levels: np.ndarray | None = None
        self._rng = np.random.default_rng(config.seed)

    def name(self) -> str:
        return self.config.compressor_pipeline

    def is_finite_grid(self) -> bool:
        return self.config.resolved_quantizer_family in {"db", "du"}

    def dynamic_rule(self) -> str:
        if not self.is_finite_grid():
            return "disabled"
        if self.config.dynamic_mode == "static":
            return "static-shared-lattice"
        if self.config.dynamic_mode == "dynamic":
            return "dynamic-shared-lattice"
        return "disabled"

    def _level_count(self) -> int:
        return max(1, 2 ** (self.config.bits_per_value - 1))

    def _build_levels(self, amax: float) -> np.ndarray:
        amax = max(float(amax), 1e-12)
        count = self._level_count()
        exponents = np.arange(count - 1, -1, -1, dtype=np.float64)
        return amax / np.power(float(self.config.p), exponents)

    @staticmethod
    def _max_abs(vectors: list[np.ndarray]) -> float:
        if not vectors:
            return 0.0
        maxima = [float(np.max(np.abs(vector))) if vector.size else 0.0 for vector in vectors]
        return max(maxima, default=0.0)

    def prepare_round(self, vectors: list[np.ndarray]) -> None:
        if not self.is_finite_grid():
            return
        observed_amax = max(self._max_abs(vectors), 1e-12)
        if self.config.dynamic_mode == "disabled":
            self._current_amax = observed_amax
        elif self._current_amax is None:
            self._current_amax = observed_amax
        elif self.config.dynamic_mode == "dynamic":
            if observed_amax > self._current_amax:
                self._current_amax = observed_amax
            else:
                while observed_amax <= self._current_amax / float(self.config.p):
                    self._current_amax /= float(self.config.p)
        self._levels = self._build_levels(self._current_amax)

    def _require_levels(self, vector: np.ndarray) -> np.ndarray:
        if self._levels is None:
            self.prepare_round([vector])
        assert self._levels is not None
        return self._levels

    def _biased_quantize(self, vector: np.ndarray) -> np.ndarray:
        levels = self._require_levels(vector)
        magnitudes = np.abs(vector)
        idx = np.abs(levels[None, :] - magnitudes[:, None]).argmin(axis=1)
        return np.sign(vector) * levels[idx]

    def _unbiased_quantize(self, vector: np.ndarray) -> np.ndarray:
        levels = self._require_levels(vector)
        magnitudes = np.abs(vector)
        upper = np.searchsorted(levels, magnitudes, side="left")
        upper = np.clip(upper, 0, len(levels) - 1)
        lower = np.maximum(upper - 1, 0)
        lower_vals = levels[lower]
        upper_vals = levels[upper]
        denom = upper_vals - lower_vals
        probs_upper = np.divide(
            magnitudes - lower_vals,
            denom,
            out=np.zeros_like(magnitudes),
            where=denom > 0,
        )
        choose_upper = self._rng.random(magnitudes.shape[0]) < probs_upper
        chosen = np.where(choose_upper, upper_vals, lower_vals)
        return np.sign(vector) * chosen

    def _topk(self, vector: np.ndarray) -> np.ndarray:
        nnz_target = max(1, int(np.ceil(self.config.k_ratio * vector.size)))
        if nnz_target >= vector.size:
            return vector.copy()
        indices = np.argpartition(np.abs(vector), -nnz_target)[-nnz_target:]
        output = np.zeros_like(vector)
        output[indices] = vector[indices]
        return output

    def _randk(self, vector: np.ndarray) -> np.ndarray:
        nnz_target = max(1, int(np.ceil(self.config.k_ratio * vector.size)))
        if nnz_target >= vector.size:
            return vector.copy()
        indices = self._rng.choice(vector.size, size=nnz_target, replace=False)
        output = np.zeros_like(vector)
        output[indices] = vector[indices]
        return output

    def compress(self, vector: np.ndarray) -> CompressionResult:
        compressed = vector.astype(np.float64, copy=True)
        state = PipelineState(
            dynamic_rule=self.dynamic_rule(),
            quantizer_family=self.config.resolved_quantizer_family,
        )
        pipeline = self.config.compressor_pipeline
        if pipeline == "fp32":
            pass
        elif pipeline == "db":
            if self.config.resolved_quantizer_family == "du":
                compressed = self._unbiased_quantize(compressed)
            else:
                compressed = self._biased_quantize(compressed)
            state.value_bits = self.config.bits_per_value
        elif pipeline == "topk":
            compressed = self._topk(compressed)
            state.sparse = True
        elif pipeline == "randk":
            compressed = self._randk(compressed)
            state.sparse = True
        elif pipeline == "db_topk":
            if self.config.resolved_quantizer_family == "du":
                compressed = self._unbiased_quantize(compressed)
            else:
                compressed = self._biased_quantize(compressed)
            compressed = self._topk(compressed)
            state.value_bits = self.config.bits_per_value
            state.sparse = True
        else:
            raise ValueError(f"Unsupported pipeline: {pipeline}")

        nonzero = int(np.count_nonzero(compressed))
        transmitted = nonzero if state.sparse else self.dimension
        bits_for_values = transmitted * state.value_bits
        bits_for_indices = 0
        if state.sparse:
            bits_for_indices = transmitted * index_bit_width(self.dimension)
        return CompressionResult(
            vector=compressed,
            stats=CompressionStats(
                transmitted_coordinates=transmitted,
                bits_for_values=bits_for_values,
                bits_for_indices=bits_for_indices,
            ),
        )


def build_pipeline(config: RunConfig, dimension: int) -> CompressionPipeline:
    return CompressionPipeline(config=config, dimension=dimension)
