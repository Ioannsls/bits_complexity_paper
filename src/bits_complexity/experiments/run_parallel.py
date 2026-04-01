from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

DEFAULT_CONFIG_PATH = Path("runs") / "configs" / "full_run_parallel.json"
POLL_INTERVAL_SEC = 0.1


@dataclass(frozen=True)
class ParallelRunConfig:
    datasets: tuple[str, ...]
    families: tuple[str, ...]
    max_parallel_tasks: int
    shutdown_timeout_sec: float
    common_args: dict[str, object]
    family_overrides: dict[str, dict[str, object]]
    task_overrides: dict[str, dict[str, object]]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    family: str
    args: dict[str, object]

    @property
    def label(self) -> str:
        return f"{self.dataset}:{self.family}"


@dataclass
class ActiveProcess:
    task: TaskSpec
    process: subprocess.Popen[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запустить orchestrator publication family-run задач по JSON-конфигу.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON-конфиг orchestrator'а.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Не запускать новые эксперименты: перед стартом сделать backup текущей папки plots "
            "и пересобрать графики из существующих metrics.csv."
        ),
    )
    return parser


def _validate_args_map(raw: object, field_name: str) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(raw)


def load_parallel_config(path: Path) -> ParallelRunConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {
        "datasets",
        "families",
        "max_parallel_tasks",
        "shutdown_timeout_sec",
        "common_args",
        "family_overrides",
    }
    missing = required_fields.difference(payload)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing config fields: {missing_str}")

    datasets = payload["datasets"]
    families = payload["families"]
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(item, str) for item in datasets)
    ):
        raise ValueError("datasets must be a non-empty list of strings")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(item, str) for item in families)
    ):
        raise ValueError("families must be a non-empty list of strings")

    max_parallel_tasks = int(payload["max_parallel_tasks"])
    shutdown_timeout_sec = float(payload["shutdown_timeout_sec"])
    if max_parallel_tasks <= 0:
        raise ValueError("max_parallel_tasks must be positive")
    if shutdown_timeout_sec < 0:
        raise ValueError("shutdown_timeout_sec must be non-negative")

    common_args = _validate_args_map(payload["common_args"], "common_args")
    family_overrides = _validate_args_map(payload["family_overrides"], "family_overrides")
    task_overrides = _validate_args_map(payload.get("task_overrides"), "task_overrides")
    validated_overrides: dict[str, dict[str, object]] = {}
    for family_name, override_payload in family_overrides.items():
        if not isinstance(family_name, str):
            raise ValueError("family_overrides keys must be strings")
        validated_overrides[family_name] = _validate_args_map(
            override_payload,
            f"family_overrides[{family_name}]",
        )
    validated_task_overrides: dict[str, dict[str, object]] = {}
    for task_name, override_payload in task_overrides.items():
        if not isinstance(task_name, str):
            raise ValueError("task_overrides keys must be strings")
        validated_task_overrides[task_name] = _validate_args_map(
            override_payload,
            f"task_overrides[{task_name}]",
        )

    return ParallelRunConfig(
        datasets=tuple(datasets),
        families=tuple(families),
        max_parallel_tasks=max_parallel_tasks,
        shutdown_timeout_sec=shutdown_timeout_sec,
        common_args=common_args,
        family_overrides=validated_overrides,
        task_overrides=validated_task_overrides,
    )


def build_task_specs(config: ParallelRunConfig) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for dataset in config.datasets:
        for family in config.families:
            args = dict(config.common_args)
            args.update(config.family_overrides.get(family, {}))
            args.update(config.task_overrides.get(f"{dataset}:{family}", {}))
            tasks.append(TaskSpec(dataset=dataset, family=family, args=args))
    return tasks


def _option_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def build_run_family_command(task: TaskSpec, *, plots_only: bool = False) -> list[str]:
    command = [sys.executable, "-m", "bits_complexity.experiments.run_family"]
    for key, value in task.args.items():
        option = _option_name(key)
        if isinstance(value, bool):
            if value:
                command.append(option)
            continue
        if value is None:
            continue
        command.extend([option, str(value)])
    if plots_only:
        command.append("--rebuild-from-existing")
    command.extend(["--dataset", task.dataset, "--family", task.family])
    return command


def _timestamp_token() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def build_plot_backup_destination(output_dir: Path, stamp: str) -> Path:
    return output_dir / "plot_backups" / f"_plot_backup_{stamp}" / "plots"


def _resolve_task_output_dir(task: TaskSpec) -> Path:
    raw = task.args.get("output_dir")
    if raw is None:
        return Path("outputs")
    return Path(str(raw))


def backup_existing_plots(tasks: list[TaskSpec], *, stamp: str | None = None) -> None:
    stamp_value = stamp or _timestamp_token()
    unique_output_dirs = sorted({_resolve_task_output_dir(task).resolve() for task in tasks})
    for output_dir in unique_output_dirs:
        plots_dir = output_dir / "plots"
        if not plots_dir.exists():
            print(f"[backup] skip backup: no plots dir at {plots_dir}")
            continue
        destination = build_plot_backup_destination(output_dir, stamp_value)
        if destination.exists():
            raise ValueError(f"Backup destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plots_dir), str(destination))
        print(f"[backup] moved {plots_dir} -> {destination}")


class ParallelOrchestrator:
    def __init__(
        self,
        config: ParallelRunConfig,
        *,
        command_builder: Callable[[TaskSpec], list[str]] = build_run_family_command,
    ) -> None:
        self.config = config
        self.command_builder = command_builder
        self._shutdown_requested = False
        self._shutdown_signal = signal.SIGINT
        self._stop_launching = False
        self._active: list[ActiveProcess] = []

    def request_shutdown(self, shutdown_signal: signal.Signals) -> None:
        self._shutdown_requested = True
        self._shutdown_signal = shutdown_signal
        self._stop_launching = True

    def _launch_task(self, task: TaskSpec) -> None:
        command = self.command_builder(task)
        print(f"[start] {task.label}")
        process = subprocess.Popen(
            command,
            start_new_session=True,
            text=True,
        )
        self._active.append(ActiveProcess(task=task, process=process))

    def _collect_finished(self) -> int | None:
        failure_code: int | None = None
        still_active: list[ActiveProcess] = []
        for active in self._active:
            return_code = active.process.poll()
            if return_code is None:
                still_active.append(active)
                continue
            print(f"[done] {active.task.label} exit={return_code}")
            if return_code != 0 and failure_code is None:
                failure_code = return_code
        self._active = still_active
        return failure_code

    def _signal_active(self, shutdown_signal: signal.Signals) -> None:
        for active in self._active:
            if active.process.poll() is not None:
                continue
            try:
                os.killpg(active.process.pid, shutdown_signal)
            except ProcessLookupError:
                continue

    def _terminate_active(self) -> None:
        if not self._active:
            return
        self._signal_active(signal.SIGINT)
        deadline = time.monotonic() + self.config.shutdown_timeout_sec
        while self._active and time.monotonic() < deadline:
            self._collect_finished()
            if self._active:
                time.sleep(POLL_INTERVAL_SEC)
        if self._active:
            self._signal_active(signal.SIGKILL)
            while self._active:
                self._collect_finished()
                if self._active:
                    time.sleep(POLL_INTERVAL_SEC)

    def run(self) -> int:
        tasks = build_task_specs(self.config)
        next_task_index = 0
        failure_code: int | None = None
        while next_task_index < len(tasks) or self._active:
            while (
                not self._stop_launching
                and next_task_index < len(tasks)
                and len(self._active) < self.config.max_parallel_tasks
            ):
                self._launch_task(tasks[next_task_index])
                next_task_index += 1

            failure_code = self._collect_finished() or failure_code
            if failure_code is not None and not self._shutdown_requested:
                self.request_shutdown(signal.SIGTERM)

            if self._shutdown_requested:
                self._terminate_active()
                return failure_code or (128 + int(self._shutdown_signal))

            if next_task_index < len(tasks) or self._active:
                time.sleep(POLL_INTERVAL_SEC)
        return 0


def _install_signal_handlers(orchestrator: ParallelOrchestrator) -> None:
    def handle_signal(signum: int, _frame: object) -> None:
        orchestrator.request_shutdown(signal.Signals(signum))

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> None:
    args = build_parser().parse_args()
    config = load_parallel_config(args.config)
    if args.plots_only:
        tasks = build_task_specs(config)
        backup_existing_plots(tasks)
        orchestrator = ParallelOrchestrator(
            config,
            command_builder=lambda task: build_run_family_command(task, plots_only=True),
        )
    else:
        orchestrator = ParallelOrchestrator(config)
    _install_signal_handlers(orchestrator)
    raise SystemExit(orchestrator.run())


if __name__ == "__main__":
    main()
