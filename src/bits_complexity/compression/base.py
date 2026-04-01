from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressionStats:
    transmitted_coordinates: int
    bits_for_values: int
    bits_for_indices: int

    @property
    def step_bits(self) -> int:
        return self.bits_for_values + self.bits_for_indices


@dataclass
class CompressionResult:
    vector: np.ndarray
    stats: CompressionStats


def index_bit_width(dimension: int) -> int:
    if dimension <= 1:
        return 0
    return int(np.ceil(np.log2(dimension)))
