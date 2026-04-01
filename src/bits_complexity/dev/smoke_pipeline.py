from __future__ import annotations

import argparse
from pathlib import Path

from bits_complexity.common.config import RunConfig
from bits_complexity.common.io import write_json
from bits_complexity.experiments.presets import family_manifest_path
from bits_complexity.experiments.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Запустить короткий end-to-end smoke pipeline на реальных данных: "
            "single run, DIANA run и упрощённый family manifest."
        ),
        epilog=(
            "Smoke pipeline намеренно короткий и budget-aware: он проверяет интеграцию, "
            "а не качество финальных кривых."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Каталог с исходными датасетами.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_smoke_final"),
        help="Каталог для smoke-артефактов.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Случайное зерно для всех smoke-запусков.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Короткая верхняя граница числа итераций smoke-run.",
    )
    parser.add_argument(
        "--cutoff-kbits-per-n",
        type=float,
        default=0.05,
        help="Короткий коммуникационный budget для smoke-проверки.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие smoke-артефакты.",
    )
    return parser


def _run_single(
    output_dir: Path, datasets_dir: Path, seed: int, max_iterations: int, cutoff: float, force: bool
) -> dict[str, str]:
    config = RunConfig(
        dataset="a9a",
        method="ef21",
        compressor_pipeline="db_topk",
        quantizer_family="db",
        dynamic_mode="static",
        bits_per_value=4,
        p=2,
        seed=seed,
        output_dir=output_dir,
        max_iterations=max_iterations,
        cutoff_kbits_per_n=cutoff,
        force=force,
    )
    artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
    return {"csv": str(artifacts.csv_path), "config": str(artifacts.config_path)}


def _run_diana(
    output_dir: Path, datasets_dir: Path, seed: int, max_iterations: int, cutoff: float, force: bool
) -> dict[str, str]:
    config = RunConfig(
        dataset="a9a",
        method="diana",
        compressor_pipeline="db",
        quantizer_family="du",
        dynamic_mode="static",
        bits_per_value=4,
        p=2,
        seed=seed,
        output_dir=output_dir,
        max_iterations=max_iterations,
        cutoff_kbits_per_n=cutoff,
        force=force,
    )
    artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
    return {"csv": str(artifacts.csv_path), "config": str(artifacts.config_path)}


def _run_family_manifest(
    output_dir: Path,
    datasets_dir: Path,
    seed: int,
    max_iterations: int,
    cutoff: float,
    force: bool,
) -> dict[str, str]:
    configs = [
        (
            "EF21 + 32-bit",
            RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="fp32",
                quantizer_family="none",
                dynamic_mode="disabled",
                bits_per_value=32,
                seed=seed,
                output_dir=output_dir,
                max_iterations=max_iterations,
                cutoff_kbits_per_n=cutoff,
                force=force,
            ),
        ),
        (
            "EF21 + Top-10%",
            RunConfig(
                dataset="a9a",
                method="ef21",
                compressor_pipeline="topk",
                quantizer_family="none",
                dynamic_mode="disabled",
                bits_per_value=32,
                seed=seed,
                output_dir=output_dir,
                max_iterations=max_iterations,
                cutoff_kbits_per_n=cutoff,
                force=force,
            ),
        ),
    ]
    entries = []
    for label, config in configs:
        artifacts = run_experiment(config=config, datasets_dir=datasets_dir)
        entries.append(
            {
                "label": label,
                "csv_path": str(artifacts.csv_path),
                "config_path": str(artifacts.config_path),
                "dataset": config.dataset,
                "family": "A-smoke",
                "method": config.method,
                "compressor_pipeline": config.compressor_pipeline,
                "quantizer_family": config.resolved_quantizer_family,
                "dynamic_mode": config.dynamic_mode,
                "bits_per_value": config.bits_per_value,
                "p": config.p,
                "seed": config.seed,
            }
        )
    manifest_path = family_manifest_path(output_dir, "a9a", "A_smoke")
    write_json(
        manifest_path,
        {
            "dataset": "a9a",
            "family": "A_smoke",
            "plot_path": str(output_dir / "plots" / "family_A_smoke_a9a.png"),
            "entries": entries,
        },
    )
    return {"manifest": str(manifest_path)}


def main() -> None:
    args = build_parser().parse_args()
    single = _run_single(
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
        seed=args.seed,
        max_iterations=args.max_iterations,
        cutoff=args.cutoff_kbits_per_n,
        force=args.force,
    )
    diana = _run_diana(
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
        seed=args.seed,
        max_iterations=args.max_iterations,
        cutoff=args.cutoff_kbits_per_n,
        force=args.force,
    )
    family = _run_family_manifest(
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
        seed=args.seed,
        max_iterations=args.max_iterations,
        cutoff=args.cutoff_kbits_per_n,
        force=args.force,
    )
    summary_path = args.output_dir / "smoke_summary.json"
    write_json(
        summary_path,
        {
            "single_run": single,
            "diana_run": diana,
            "family_run": family,
        },
    )
    print(summary_path)


if __name__ == "__main__":
    main()
