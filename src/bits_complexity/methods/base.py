from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Union

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.pipelines import CompressionPipeline
from bits_complexity.problems.logistic import LogisticProblem

CSV_FIELDS = [
    "seed",
    "dataset",
    "method",
    "compressor_pipeline",
    "quantizer_family",
    "dynamic_mode",
    "p",
    "bits_per_value",
    "k",
    "k_ratio",
    "iteration",
    "transmitted_coordinates",
    "bits_for_values",
    "bits_for_indices",
    "step_bits",
    "cum_bits",
    "kbits_per_n",
    "objective",
    "grad_norm_sq",
    "accuracy",
    "runtime_sec",
]


@dataclass
class IterationRecord:
    iteration: int
    transmitted_coordinates: int
    bits_for_values: int
    bits_for_indices: int
    step_bits: int
    cum_bits: int
    kbits_per_n: float
    objective: float
    grad_norm_sq: float
    accuracy: float
    runtime_sec: float

    def to_csv_row(self, config: RunConfig, k: int) -> Dict[str, Union[float, int, str]]:
        return {
            "seed": config.seed,
            "dataset": config.dataset,
            "method": config.method,
            "compressor_pipeline": config.compressor_pipeline,
            "quantizer_family": config.resolved_quantizer_family,
            "dynamic_mode": config.dynamic_mode,
            "p": config.p,
            "bits_per_value": config.bits_per_value,
            "k": k,
            "k_ratio": config.k_ratio,
            "iteration": self.iteration,
            "transmitted_coordinates": self.transmitted_coordinates,
            "bits_for_values": self.bits_for_values,
            "bits_for_indices": self.bits_for_indices,
            "step_bits": self.step_bits,
            "cum_bits": self.cum_bits,
            "kbits_per_n": self.kbits_per_n,
            "objective": self.objective,
            "grad_norm_sq": self.grad_norm_sq,
            "accuracy": self.accuracy,
            "runtime_sec": self.runtime_sec,
        }


def build_initial_record(config: RunConfig, problem: LogisticProblem) -> IterationRecord:
    weights = problem.initial_point()
    return IterationRecord(
        iteration=0,
        transmitted_coordinates=0,
        bits_for_values=0,
        bits_for_indices=0,
        step_bits=0,
        cum_bits=0,
        kbits_per_n=0.0,
        objective=problem.objective(weights),
        grad_norm_sq=problem.grad_norm_sq(weights),
        accuracy=problem.accuracy(weights),
        runtime_sec=0.0,
    )


class TrainingMethod:
    def __init__(
        self,
        config: RunConfig,
        problem: LogisticProblem,
        client_data: list[tuple],
        pipeline: CompressionPipeline,
    ) -> None:
        self.config = config
        self.problem = problem
        self.client_data = client_data
        self.pipeline = pipeline
        self.dimension = problem.dimension
        self.weights = problem.initial_point()
        self.cum_bits = 0
        self.start_time = perf_counter()

    def _objective(self) -> float:
        return self.problem.objective(self.weights)

    def _grad_norm_sq(self) -> float:
        return self.problem.grad_norm_sq(self.weights)

    def _accuracy(self) -> float:
        return self.problem.accuracy(self.weights)

    def _runtime(self) -> float:
        return perf_counter() - self.start_time

    def _kbits_per_n(self) -> float:
        return self.cum_bits / 1000.0 / self.dimension

    def step(self, iteration: int) -> IterationRecord:
        raise NotImplementedError

    def run(self) -> list[IterationRecord]:
        records: list[IterationRecord] = []
        for iteration in range(1, self.config.max_iterations + 1):
            record = self.step(iteration)
            records.append(record)
            if record.kbits_per_n >= self.config.cutoff_kbits_per_n:
                break
        return records
