from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from bits_complexity.common.config import RunConfig, validate_config
from bits_complexity.common.io import ensure_dir, write_csv, write_json
from bits_complexity.compression.pipelines import build_pipeline
from bits_complexity.data.datasets import load_dataset, split_into_clients
from bits_complexity.methods.base import CSV_FIELDS, build_initial_record
from bits_complexity.methods.factory import build_method
from bits_complexity.plots.builder import plot_run_debug_bundle
from bits_complexity.problems.logistic import LogisticProblem


@dataclass
class RunArtifacts:
    csv_path: Path
    config_path: Path
    reused_existing: bool


@dataclass
class LearningRateResolution:
    requested_learning_rate: float
    base_resolved_learning_rate: float
    resolved_learning_rate: float
    multiplier: float
    auto_enabled: bool
    search_iterations: int
    smoothness_constant: float | None = None
    trial_runs: int = 0


DEFAULT_LR_SEARCH_ITERATIONS = 128
LR_SEARCH_MAX_BRACKET_STEPS = 20
LR_SEARCH_BINARY_STEPS = 25
LR_SEARCH_TOLERANCE = 1e-12
LR_SEARCH_SAFETY_SHRINK: float = 1 / 4


def _paths_for_run(config: RunConfig, resolved_learning_rate: float) -> tuple[Path, Path]:
    run_dir = ensure_dir(
        config.output_dir
        / "runs"
        / config.dataset
        / config.run_slug_with_learning_rate(resolved_learning_rate=resolved_learning_rate)
    )
    return run_dir / "metrics.csv", run_dir / "config.json"


def _trial_config(config: RunConfig, learning_rate: float) -> RunConfig:
    return replace(config, learning_rate=learning_rate, learning_rate_auto=False)


def _is_monotone_trial(
    config: RunConfig,
    problem: LogisticProblem,
    client_data: list[tuple],
    trial_iterations: int,
) -> bool:
    pipeline = build_pipeline(config=config, dimension=problem.dimension)
    method = build_method(
        config=config, problem=problem, client_data=client_data, pipeline=pipeline
    )
    previous_value = problem.grad_norm_sq(problem.initial_point())
    for iteration in range(1, trial_iterations + 1):
        record = method.step(iteration)
        tolerance = LR_SEARCH_TOLERANCE * max(1.0, previous_value)
        if record.grad_norm_sq > previous_value + tolerance:
            return False
        previous_value = record.grad_norm_sq
    return True


def resolve_learning_rate(
    config: RunConfig,
    problem: LogisticProblem,
    client_data: list[tuple],
) -> LearningRateResolution:
    if not config.learning_rate_auto:
        return LearningRateResolution(
            requested_learning_rate=config.learning_rate,
            base_resolved_learning_rate=config.learning_rate,
            resolved_learning_rate=config.learning_rate * config.learning_rate_multiplier,
            multiplier=config.learning_rate_multiplier,
            auto_enabled=False,
            search_iterations=0,
        )

    smoothness_constant = max(problem.smoothness_constant(), 1e-12)
    trial_iterations = DEFAULT_LR_SEARCH_ITERATIONS
    trial_runs = 0

    def passes(learning_rate: float) -> bool:
        nonlocal trial_runs
        trial_runs += 1
        return _is_monotone_trial(
            config=_trial_config(config, learning_rate),
            problem=problem,
            client_data=client_data,
            trial_iterations=trial_iterations,
        )

    start_lr = 1.0 / smoothness_constant
    safety_cap = start_lr * (2**LR_SEARCH_MAX_BRACKET_STEPS)

    if passes(start_lr):
        low = start_lr
        high = start_lr
        found_failure = False
        for _ in range(LR_SEARCH_MAX_BRACKET_STEPS):
            candidate = high * 2.0
            if candidate > safety_cap:
                break
            if passes(candidate):
                low = candidate
                high = candidate
                continue
            high = candidate
            found_failure = True
            break
        monotone_learning_rate = low if found_failure else high
    else:
        high = start_lr
        low = 0.0
        for _ in range(LR_SEARCH_MAX_BRACKET_STEPS):
            candidate = high / 2.0
            if passes(candidate):
                low = candidate
                break
            high = candidate
        for _ in range(LR_SEARCH_BINARY_STEPS):
            midpoint = 0.5 * (low + high)
            if passes(midpoint):
                low = midpoint
            else:
                high = midpoint
        monotone_learning_rate = low

    base_resolved_learning_rate = LR_SEARCH_SAFETY_SHRINK * monotone_learning_rate
    resolved_learning_rate = base_resolved_learning_rate * config.learning_rate_multiplier

    return LearningRateResolution(
        requested_learning_rate=config.learning_rate,
        base_resolved_learning_rate=base_resolved_learning_rate,
        resolved_learning_rate=resolved_learning_rate,
        multiplier=config.learning_rate_multiplier,
        auto_enabled=True,
        search_iterations=trial_iterations,
        smoothness_constant=smoothness_constant,
        trial_runs=trial_runs,
    )


def run_experiment(config: RunConfig, datasets_dir: Path) -> RunArtifacts:
    validate_config(config)
    bundle = load_dataset(name=config.dataset, datasets_dir=datasets_dir, seed=config.seed)
    client_data = split_into_clients(bundle.train_features, bundle.train_labels, config.clients)
    problem = LogisticProblem(
        train_features=bundle.train_features,
        train_labels=bundle.train_labels,
        test_features=bundle.test_features,
        test_labels=bundle.test_labels,
        l2_reg=config.l2_reg,
    )
    lr_resolution = resolve_learning_rate(config=config, problem=problem, client_data=client_data)
    effective_config = replace(
        config,
        learning_rate=lr_resolution.resolved_learning_rate,
        learning_rate_auto=False,
    )
    csv_path, config_path = _paths_for_run(config, lr_resolution.resolved_learning_rate)
    if csv_path.exists() and config_path.exists() and not config.force:
        return RunArtifacts(csv_path=csv_path, config_path=config_path, reused_existing=True)

    pipeline = build_pipeline(config=config, dimension=problem.dimension)

    if config.dry_run:
        write_json(
            config_path,
            {
                **config.to_dict(),
                "learning_rate_auto": lr_resolution.auto_enabled,
                "learning_rate_multiplier": lr_resolution.multiplier,
                "base_resolved_learning_rate": lr_resolution.base_resolved_learning_rate,
                "resolved_learning_rate": lr_resolution.resolved_learning_rate,
                "learning_rate_search_iterations": lr_resolution.search_iterations,
                "learning_rate_search_smoothness_constant": lr_resolution.smoothness_constant,
                "learning_rate_search_trial_runs": lr_resolution.trial_runs,
                "feature_dim": problem.dimension,
                "train_size": bundle.train_size,
                "dynamic_rule": pipeline.dynamic_rule(),
                "dry_run": True,
            },
        )
        write_csv(csv_path, CSV_FIELDS, [])
        return RunArtifacts(csv_path=csv_path, config_path=config_path, reused_existing=False)

    records = [build_initial_record(config=config, problem=problem)]
    pipeline = build_pipeline(config=effective_config, dimension=problem.dimension)
    method = build_method(
        config=effective_config,
        problem=problem,
        client_data=client_data,
        pipeline=pipeline,
    )
    records.extend(method.run())
    k = max(1, int(round(config.k_ratio * problem.dimension)))
    rows = [record.to_csv_row(config=effective_config, k=k) for record in records]
    write_csv(csv_path, CSV_FIELDS, rows)
    write_json(
        config_path,
        {
            **config.to_dict(),
            "learning_rate_auto": lr_resolution.auto_enabled,
            "learning_rate_multiplier": lr_resolution.multiplier,
            "base_resolved_learning_rate": lr_resolution.base_resolved_learning_rate,
            "resolved_learning_rate": lr_resolution.resolved_learning_rate,
            "learning_rate_search_iterations": lr_resolution.search_iterations,
            "learning_rate_search_smoothness_constant": lr_resolution.smoothness_constant,
            "learning_rate_search_trial_runs": lr_resolution.trial_runs,
            "feature_dim": problem.dimension,
            "train_size": bundle.train_size,
            "dynamic_rule": pipeline.dynamic_rule(),
            "records": len(rows),
        },
    )
    if config.plot:
        plot_run_debug_bundle(csv_path, config.output_dir / "plots" / "debug")
    return RunArtifacts(csv_path=csv_path, config_path=config_path, reused_existing=False)
