from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET_PACKAGE = ROOT / "src" / "bits_complexity"
DEFAULT_COVERAGE_THRESHOLD = 90.0
EXCLUDED_COVERAGE_PARTS = {"dev"}


def _run_subprocess(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def run_lint() -> int:
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        print("ruff is not installed. Install dev dependencies first.", file=sys.stderr)
        return 1
    commands = [
        [ruff_bin, "check", str(ROOT)],
        [ruff_bin, "format", "--check", str(ROOT)],
    ]
    for command in commands:
        code = _run_subprocess(command)
        if code != 0:
            return code
    return 0


def run_tests() -> int:
    return _run_subprocess([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def _parse_trace_summary(output: str) -> tuple[float, int, int]:
    total_lines = 0
    covered_lines = 0
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)%\s+(\S+)\s+\((.+)\)$")
    for raw_line in output.splitlines():
        match = pattern.match(raw_line)
        if not match:
            continue
        executable_lines = int(match.group(1))
        coverage_pct = int(match.group(2))
        path = Path(match.group(4)).resolve()
        if not str(path).startswith(str(TARGET_PACKAGE)):
            continue
        if any(part in EXCLUDED_COVERAGE_PARTS for part in path.parts):
            continue
        total_lines += executable_lines
        covered_lines += round(executable_lines * coverage_pct / 100.0)
    percentage = 100.0 if total_lines == 0 else covered_lines * 100.0 / total_lines
    return percentage, covered_lines, total_lines


def run_coverage(threshold: float) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = [
            sys.executable,
            "-m",
            "trace",
            "--count",
            "--summary",
            "--coverdir",
            tmp_dir,
            "--module",
            "unittest",
            "discover",
            "-s",
            "tests",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return int(completed.returncode)
    sys.stdout.write(completed.stdout)
    percentage, covered, total = _parse_trace_summary(completed.stdout)
    print(f"Coverage: {percentage:.2f}% ({covered}/{total} executable lines)")
    if percentage < threshold:
        print(
            f"Coverage gate failed: required {threshold:.2f}%, got {percentage:.2f}%.",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Запустить quality gate для small-datasets проекта: lint, tests, coverage "
            "или полный набор проверок."
        ),
    )
    parser.add_argument(
        "command",
        choices=("lint", "test", "coverage", "all"),
        help="Подкоманда quality gate.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help="Минимально допустимое покрытие рабочих модулей, в процентах.",
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Пропустить ruff-проверки в окружениях без dev-зависимостей.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "lint":
        raise SystemExit(run_lint())
    if args.command == "test":
        raise SystemExit(run_tests())
    if args.command == "coverage":
        raise SystemExit(run_coverage(args.coverage_threshold))

    if not args.skip_lint:
        lint_code = run_lint()
        if lint_code != 0:
            raise SystemExit(lint_code)
    test_code = run_tests()
    if test_code != 0:
        raise SystemExit(test_code)
    raise SystemExit(run_coverage(args.coverage_threshold))


if __name__ == "__main__":
    main()
