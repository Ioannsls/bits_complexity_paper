from __future__ import annotations

from bits_complexity.common.config import RunConfig
from bits_complexity.compression.pipelines import CompressionPipeline
from bits_complexity.methods.diana import build_diana
from bits_complexity.methods.ef21 import build_ef21
from bits_complexity.problems.logistic import LogisticProblem


def build_method(
    config: RunConfig,
    problem: LogisticProblem,
    client_data: list[tuple],
    pipeline: CompressionPipeline,
):
    if config.method == "ef21":
        return build_ef21(
            config=config, problem=problem, client_data=client_data, pipeline=pipeline
        )
    if config.method == "diana":
        return build_diana(
            config=config, problem=problem, client_data=client_data, pipeline=pipeline
        )
    raise ValueError(f"Unsupported method: {config.method}")
