from __future__ import annotations

import numpy as np

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.pipelines import CompressionPipeline
from bits_complexity.methods.base import IterationRecord, TrainingMethod
from bits_complexity.problems.logistic import LogisticProblem


class EF21Method(TrainingMethod):
    def __init__(
        self,
        config: RunConfig,
        problem: LogisticProblem,
        client_data: list[tuple[np.ndarray, np.ndarray]],
        pipeline: CompressionPipeline,
    ) -> None:
        super().__init__(config=config, problem=problem, client_data=client_data, pipeline=pipeline)
        self.memory = [np.zeros(self.dimension, dtype=np.float64) for _ in client_data]
        self.gradient_estimate = np.zeros(self.dimension, dtype=np.float64)
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        gradients = [
            self.problem.gradient(self.weights, features, labels)
            for features, labels in self.client_data
        ]
        self.pipeline.prepare_round(gradients)
        messages = []
        total_step_bits = 0
        for index, gradient in enumerate(gradients):
            result = self.pipeline.compress(gradient)
            self.memory[index] = result.vector
            messages.append(result.vector)
            total_step_bits += result.stats.step_bits
        self.gradient_estimate = np.mean(np.stack(messages, axis=0), axis=0)
        self.cum_bits += total_step_bits

    def step(self, iteration: int) -> IterationRecord:
        self.weights = self.weights - self.config.learning_rate * self.gradient_estimate
        deltas = []
        for index, (features, labels) in enumerate(self.client_data):
            local_gradient = self.problem.gradient(self.weights, features, labels)
            deltas.append(local_gradient - self.memory[index])
        self.pipeline.prepare_round(deltas)

        transmitted_coordinates = 0
        bits_for_values = 0
        bits_for_indices = 0
        for index, delta in enumerate(deltas):
            result = self.pipeline.compress(delta)
            self.memory[index] = self.memory[index] + result.vector
            transmitted_coordinates += result.stats.transmitted_coordinates
            bits_for_values += result.stats.bits_for_values
            bits_for_indices += result.stats.bits_for_indices
        self.gradient_estimate = np.mean(np.stack(self.memory, axis=0), axis=0)
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


def build_ef21(
    config: RunConfig,
    problem: LogisticProblem,
    client_data: list[tuple[np.ndarray, np.ndarray]],
    pipeline: CompressionPipeline,
) -> EF21Method:
    return EF21Method(config=config, problem=problem, client_data=client_data, pipeline=pipeline)
