from __future__ import annotations

import numpy as np

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.pipelines import CompressionPipeline
from bits_complexity.methods.base import IterationRecord, TrainingMethod
from bits_complexity.problems.logistic import LogisticProblem


class DianaMethod(TrainingMethod):
    def __init__(
        self,
        config: RunConfig,
        problem: LogisticProblem,
        client_data: list[tuple[np.ndarray, np.ndarray]],
        pipeline: CompressionPipeline,
    ) -> None:
        super().__init__(config=config, problem=problem, client_data=client_data, pipeline=pipeline)
        self.shifts = [np.zeros(self.dimension, dtype=np.float64) for _ in client_data]
        self.global_shift = np.zeros(self.dimension, dtype=np.float64)

    def step(self, iteration: int) -> IterationRecord:
        deltas = []
        for index, (features, labels) in enumerate(self.client_data):
            gradient = self.problem.gradient(self.weights, features, labels)
            deltas.append(gradient - self.shifts[index])
        self.pipeline.prepare_round(deltas)

        compressed_deltas = []
        transmitted_coordinates = 0
        bits_for_values = 0
        bits_for_indices = 0
        for delta in deltas:
            result = self.pipeline.compress(delta)
            compressed_deltas.append(result.vector)
            transmitted_coordinates += result.stats.transmitted_coordinates
            bits_for_values += result.stats.bits_for_values
            bits_for_indices += result.stats.bits_for_indices

        gradient_estimate = self.global_shift + np.mean(np.stack(compressed_deltas, axis=0), axis=0)
        self.weights = self.weights - self.config.learning_rate * gradient_estimate

        for index, compressed_delta in enumerate(compressed_deltas):
            self.shifts[index] = self.shifts[index] + self.config.diana_alpha * compressed_delta
        self.global_shift = np.mean(np.stack(self.shifts, axis=0), axis=0)

        step_bits = bits_for_values + bits_for_indices
        self.cum_bits += step_bits
        return IterationRecord(
            iteration=iteration,
            transmitted_coordinates=transmitted_coordinates,
            bits_for_values=bits_for_values,
            bits_for_indices=bits_for_indices,
            step_bits=step_bits,
            cum_bits=self.cum_bits,
            kbits_per_n=self._kbits_per_n(),
            objective=self._objective(),
            grad_norm_sq=self._grad_norm_sq(),
            accuracy=self._accuracy(),
            runtime_sec=self._runtime(),
        )


def build_diana(
    config: RunConfig,
    problem: LogisticProblem,
    client_data: list[tuple[np.ndarray, np.ndarray]],
    pipeline: CompressionPipeline,
) -> DianaMethod:
    return DianaMethod(config=config, problem=problem, client_data=client_data, pipeline=pipeline)
