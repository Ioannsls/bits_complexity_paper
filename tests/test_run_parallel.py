from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bits_complexity.experiments.run_parallel import (
    DEFAULT_CONFIG_PATH,
    ParallelOrchestrator,
    ParallelRunConfig,
    TaskSpec,
    backup_existing_plots,
    build_plot_backup_destination,
    build_run_family_command,
    build_task_specs,
    load_parallel_config,
)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_file(path: Path, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for file: {path}")


class ParallelRunnerTest(unittest.TestCase):
    def test_load_parallel_config_requires_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(json.dumps({"datasets": ["a9a"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing config fields"):
                load_parallel_config(path)

    def test_load_parallel_config_and_build_tasks_merge_family_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": ["a9a", "w8a"],
                        "families": ["db", "du"],
                        "max_parallel_tasks": 2,
                        "shutdown_timeout_sec": 1.0,
                        "common_args": {"output_dir": "outputs", "force": True},
                        "family_overrides": {"du": {"cutoff_kbits_per_n": 77}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_parallel_config(path)
            tasks = build_task_specs(config)
            self.assertEqual(len(tasks), 4)
            self.assertEqual(tasks[0].dataset, "a9a")
            self.assertEqual(tasks[0].family, "db")
            self.assertEqual(tasks[0].args["output_dir"], "outputs")
            self.assertNotIn("cutoff_kbits_per_n", tasks[0].args)
            self.assertEqual(tasks[1].family, "du")
            self.assertEqual(tasks[1].args["cutoff_kbits_per_n"], 77)

    def test_load_parallel_config_and_build_tasks_apply_task_override_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": ["a9a", "w8a"],
                        "families": ["db_topk"],
                        "max_parallel_tasks": 2,
                        "shutdown_timeout_sec": 1.0,
                        "common_args": {"cutoff_kbits_per_n": 200, "force": True},
                        "family_overrides": {"db_topk": {"cutoff_kbits_per_n": 100}},
                        "task_overrides": {"w8a:db_topk": {"cutoff_kbits_per_n": 10}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_parallel_config(path)
            tasks = build_task_specs(config)
            self.assertEqual(tasks[0].dataset, "a9a")
            self.assertEqual(tasks[0].args["cutoff_kbits_per_n"], 100)
            self.assertEqual(tasks[1].dataset, "w8a")
            self.assertEqual(tasks[1].args["cutoff_kbits_per_n"], 10)

    def test_build_run_family_command_serializes_args(self) -> None:
        task = TaskSpec(
            dataset="a9a",
            family="db",
            args={
                "output_dir": "outputs",
                "force": True,
                "auto_learning_rate": True,
                "learning_rate_multiplier": 2.0,
                "cutoff_kbits_per_n": 50,
                "ignored": None,
            },
        )
        command = build_run_family_command(task)
        self.assertEqual(
            command[:3], [sys.executable, "-m", "bits_complexity.experiments.run_family"]
        )
        self.assertIn("--output-dir", command)
        self.assertIn("outputs", command)
        self.assertIn("--force", command)
        self.assertIn("--auto-learning-rate", command)
        self.assertIn("--learning-rate-multiplier", command)
        self.assertIn("2.0", command)
        self.assertIn("--cutoff-kbits-per-n", command)
        self.assertEqual(command[-4:], ["--dataset", "a9a", "--family", "db"])

    def test_build_run_family_command_adds_rebuild_flag_in_plots_only_mode(self) -> None:
        task = TaskSpec(dataset="a9a", family="db", args={"output_dir": "outputs"})
        command = build_run_family_command(task, plots_only=True)
        self.assertIn("--rebuild-from-existing", command)
        self.assertEqual(command[-4:], ["--dataset", "a9a", "--family", "db"])

    def test_build_plot_backup_destination_matches_contract(self) -> None:
        output_dir = Path("/tmp/results")
        backup = build_plot_backup_destination(output_dir, "20260331_123259")
        self.assertEqual(
            backup,
            Path("/tmp/results/plot_backups/_plot_backup_20260331_123259/plots"),
        )

    def test_backup_existing_plots_moves_plots_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True)
            (plots_dir / "old_plot.png").write_text("plot", encoding="utf-8")
            tasks = [TaskSpec(dataset="a9a", family="db", args={"output_dir": str(output_dir)})]
            backup_existing_plots(tasks, stamp="20260331_123259")
            backup_dir = output_dir / "plot_backups" / "_plot_backup_20260331_123259" / "plots"
            self.assertFalse(plots_dir.exists())
            self.assertTrue((backup_dir / "old_plot.png").exists())

    def test_backup_existing_plots_skips_when_plots_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            output_dir.mkdir(parents=True)
            tasks = [TaskSpec(dataset="a9a", family="db", args={"output_dir": str(output_dir)})]
            backup_existing_plots(tasks, stamp="20260331_123259")
            self.assertFalse((output_dir / "plot_backups").exists())

    def test_orchestrator_shutdown_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            grandchild_pid_path = Path(tmp_dir) / "grandchild.pid"
            config = ParallelRunConfig(
                datasets=("a9a",),
                families=("db",),
                max_parallel_tasks=1,
                shutdown_timeout_sec=1.0,
                common_args={},
                family_overrides={},
                task_overrides={},
            )

            def command_builder(_task: TaskSpec) -> list[str]:
                script = """
import pathlib, signal, subprocess, sys, time
pid_path = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pid_path.write_text(str(child.pid), encoding="utf-8")
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
while True:
    time.sleep(1)
"""
                return [sys.executable, "-c", script, str(grandchild_pid_path)]

            orchestrator = ParallelOrchestrator(config, command_builder=command_builder)
            result: dict[str, int] = {}
            thread = threading.Thread(target=lambda: result.setdefault("code", orchestrator.run()))
            thread.start()
            _wait_for_file(grandchild_pid_path)
            grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
            time.sleep(0.2)
            orchestrator.request_shutdown(signal.SIGINT)
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result["code"], 130)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and _pid_exists(grandchild_pid):
                time.sleep(0.05)
            self.assertFalse(_pid_exists(grandchild_pid))

    def test_orchestrator_fail_fast_stops_remaining_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            grandchild_pid_path = Path(tmp_dir) / "grandchild.pid"
            config = ParallelRunConfig(
                datasets=("a9a",),
                families=("db", "du"),
                max_parallel_tasks=2,
                shutdown_timeout_sec=1.0,
                common_args={},
                family_overrides={},
                task_overrides={},
            )

            def command_builder(task: TaskSpec) -> list[str]:
                if task.family == "db":
                    return [sys.executable, "-c", "import sys; sys.exit(1)"]
                script = """
import pathlib, signal, subprocess, sys, time
pid_path = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pid_path.write_text(str(child.pid), encoding="utf-8")
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
while True:
    time.sleep(1)
"""
                return [sys.executable, "-c", script, str(grandchild_pid_path)]

            orchestrator = ParallelOrchestrator(config, command_builder=command_builder)
            _wait_thread_result: dict[str, int] = {}
            thread = threading.Thread(
                target=lambda: _wait_thread_result.setdefault("code", orchestrator.run())
            )
            thread.start()
            _wait_for_file(grandchild_pid_path)
            grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(_wait_thread_result["code"], 1)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and _pid_exists(grandchild_pid):
                time.sleep(0.05)
            self.assertFalse(_pid_exists(grandchild_pid))

    def test_default_config_path_points_to_runs_configs(self) -> None:
        self.assertEqual(DEFAULT_CONFIG_PATH, Path("runs") / "configs" / "full_run_parallel.json")


if __name__ == "__main__":
    unittest.main()
