from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from bits_complexity.common.config import RunConfig
from bits_complexity.common.io import write_json
from bits_complexity.experiments.presets import (
    SUPPORTED_PUBLICATION_FAMILIES,
    family_manifest_path,
    family_plot_path,
    family_plot_x_max,
    publication_family_configs,
    publication_family_title,
)
from bits_complexity.experiments.runner import run_experiment
from bits_complexity.plots.builder import plot_family_from_manifest


@dataclass(frozen=True)
class ExistingRunArtifact:
    csv_path: Path
    config_path: Path
    payload: dict[str, object]
    records: int
    config_mtime: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Запустить предопределённое семейство экспериментов и собрать manifest "
            "для publication plot."
        ),
        epilog=(
            "Поддерживаются family: db, du, db_topk, du_topk, "
            "static_dynamic, static_dynamic_b3, static_dynamic_du и static_dynamic_du_b3. "
            "Первые четыре дают по 5 кривых (reference + 3/4/5/6-bit), "
            "семейства static_dynamic дают по 6 кривых для фиксированного b и p in {2, 4, 8}."
        ),
    )
    parser.add_argument("--dataset", required=True, help="Имя датасета для всей семьи.")
    parser.add_argument(
        "--family",
        choices=SUPPORTED_PUBLICATION_FAMILIES,
        required=True,
        help="Идентификатор семейства пресетов.",
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
        default=Path("outputs"),
        help="Корневой каталог для запусков, manifest и графиков.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Верхняя граница числа итераций для каждого запуска семьи.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Количество клиентов после разбиения train-выборки для каждого запуска семьи.",
    )
    parser.add_argument(
        "--cutoff-kbits-per-n",
        type=float,
        default=5.0,
        help="Коммуникационный бюджет остановки для каждого запуска семьи.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Случайное зерно, применяемое ко всем запускам семьи.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие артефакты family-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Создать config и пустой CSV для каждого запуска семьи без исполнения метода.",
    )
    parser.add_argument(
        "--auto-learning-rate",
        action="store_true",
        help="Включить автоподбор lr для каждого запуска внутри family-run.",
    )
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Множитель, домножаемый на итоговый effective learning rate для каждого запуска.",
    )
    parser.add_argument(
        "--rebuild-from-existing",
        action="store_true",
        help=(
            "Пересобрать manifest/plot только из уже существующих runs/<dataset>/* "
            "без запуска новых экспериментов."
        ),
    )
    return parser


def _with_overrides(
    configs: list[tuple[str, RunConfig]], args: argparse.Namespace
) -> list[tuple[str, RunConfig]]:
    updated = []
    for label, config in configs:
        cutoff = args.cutoff_kbits_per_n
        if (
            args.family in {"static_dynamic_b3", "static_dynamic_du_b3"}
            and config.bits_per_value == 3
        ):
            cutoff *= 0.75
        updated.append(
            (
                label,
                RunConfig(
                    **{
                        **config.to_dict(),
                        "clients": args.clients,
                        "output_dir": args.output_dir,
                        "max_iterations": args.max_iterations,
                        "cutoff_kbits_per_n": cutoff,
                        "seed": args.seed,
                        "force": args.force,
                        "dry_run": args.dry_run,
                        "learning_rate_auto": args.auto_learning_rate,
                        "learning_rate_multiplier": args.learning_rate_multiplier,
                    }
                ),
            )
        )
    return updated


def _as_int(value: object, *, field_name: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field '{field_name}' in {path}: {value!r}") from exc


def _as_float(value: object, *, field_name: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float field '{field_name}' in {path}: {value!r}") from exc


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _resolved_quantizer_from_payload(payload: dict[str, object]) -> str:
    resolved = payload.get("resolved_quantizer_family")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    raw = payload.get("quantizer_family")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "none"


def _discover_existing_run_artifacts(output_dir: Path, dataset: str) -> list[ExistingRunArtifact]:
    dataset_runs_dir = output_dir / "runs" / dataset
    if not dataset_runs_dir.exists():
        return []

    artifacts: list[ExistingRunArtifact] = []
    for run_dir in sorted(path for path in dataset_runs_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        csv_path = run_dir / "metrics.csv"
        if not config_path.exists() or not csv_path.exists():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        records_value = payload.get("records")
        records = (
            _count_csv_rows(csv_path)
            if records_value is None
            else _as_int(
                records_value,
                field_name="records",
                path=config_path,
            )
        )
        artifacts.append(
            ExistingRunArtifact(
                csv_path=csv_path,
                config_path=config_path,
                payload=payload,
                records=records,
                config_mtime=config_path.stat().st_mtime,
            )
        )
    return artifacts


def _matches_expected_config(expected: RunConfig, artifact: ExistingRunArtifact) -> bool:
    payload = artifact.payload
    config_path = artifact.config_path
    if str(payload.get("dataset")) != expected.dataset:
        return False
    if str(payload.get("method")) != expected.method:
        return False
    if str(payload.get("compressor_pipeline")) != expected.compressor_pipeline:
        return False
    if str(payload.get("dynamic_mode")) != expected.dynamic_mode:
        return False
    if (
        _as_int(payload.get("bits_per_value"), field_name="bits_per_value", path=config_path)
        != expected.bits_per_value
    ):
        return False
    if _as_int(payload.get("p"), field_name="p", path=config_path) != expected.p:
        return False
    if _as_int(payload.get("seed"), field_name="seed", path=config_path) != expected.seed:
        return False
    observed_k_ratio = _as_float(payload.get("k_ratio"), field_name="k_ratio", path=config_path)
    if not math.isclose(observed_k_ratio, expected.k_ratio, rel_tol=0.0, abs_tol=1e-12):
        return False
    if _resolved_quantizer_from_payload(payload) != expected.resolved_quantizer_family:
        return False
    return True


def _missing_config_description(config: RunConfig) -> str:
    parts = [
        f"dataset={config.dataset}",
        f"method={config.method}",
        f"compressor_pipeline={config.compressor_pipeline}",
        f"quantizer_family={config.resolved_quantizer_family}",
        f"dynamic_mode={config.dynamic_mode}",
        f"bits_per_value={config.bits_per_value}",
        f"p={config.p}",
        f"k_ratio={config.k_ratio:.3f}",
        f"seed={config.seed}",
    ]
    return ", ".join(parts)


def _select_best_existing_artifact(matches: list[ExistingRunArtifact]) -> ExistingRunArtifact:
    return sorted(
        matches,
        key=lambda item: (item.records, item.config_mtime, str(item.csv_path)),
        reverse=True,
    )[0]


def _build_manifest_entries_from_existing(
    configs: list[tuple[str, RunConfig]],
    *,
    output_dir: Path,
    dataset: str,
    family: str,
) -> list[dict[str, object]]:
    discovered = _discover_existing_run_artifacts(output_dir=output_dir, dataset=dataset)
    if not discovered:
        raise ValueError(f"No existing runs found in {output_dir / 'runs' / dataset}")

    missing: list[str] = []
    entries: list[dict[str, object]] = []
    for label, config in configs:
        matches = [
            artifact for artifact in discovered if _matches_expected_config(config, artifact)
        ]
        if not matches:
            missing.append(f"{label}: {_missing_config_description(config)}")
            continue
        selected = _select_best_existing_artifact(matches)
        entries.append(
            {
                "label": label,
                "csv_path": str(selected.csv_path),
                "config_path": str(selected.config_path),
                "dataset": config.dataset,
                "family": family,
                "method": config.method,
                "compressor_pipeline": config.compressor_pipeline,
                "quantizer_family": config.resolved_quantizer_family,
                "dynamic_mode": config.dynamic_mode,
                "bits_per_value": config.bits_per_value,
                "p": config.p,
                "seed": config.seed,
            }
        )
        print(selected.csv_path)
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise ValueError(
            f"Missing existing runs for family={family}, dataset={dataset}:\n{details}"
        )
    return entries


def main() -> None:
    args = build_parser().parse_args()
    configs = _with_overrides(
        publication_family_configs(
            args.dataset,
            args.output_dir,
            args.family,
            cutoff_kbits_per_n=args.cutoff_kbits_per_n,
        ),
        args,
    )
    if args.rebuild_from_existing:
        manifest_entries = _build_manifest_entries_from_existing(
            configs,
            output_dir=args.output_dir,
            dataset=args.dataset,
            family=args.family,
        )
    else:
        manifest_entries = []
        for label, config in configs:
            artifacts = run_experiment(config=config, datasets_dir=args.datasets_dir)
            manifest_entries.append(
                {
                    "label": label,
                    "csv_path": str(artifacts.csv_path),
                    "config_path": str(artifacts.config_path),
                    "dataset": config.dataset,
                    "family": args.family,
                    "method": config.method,
                    "compressor_pipeline": config.compressor_pipeline,
                    "quantizer_family": config.resolved_quantizer_family,
                    "dynamic_mode": config.dynamic_mode,
                    "bits_per_value": config.bits_per_value,
                    "p": config.p,
                    "seed": config.seed,
                }
            )
            print(artifacts.csv_path)
    manifest_path = family_manifest_path(args.output_dir, args.dataset, args.family)
    grad_plot_path = family_plot_path(
        args.output_dir,
        args.dataset,
        args.family,
        y_key="grad_norm_sq",
    )
    objective_plot_path = family_plot_path(
        args.output_dir,
        args.dataset,
        args.family,
        y_key="objective",
    )
    x_max = family_plot_x_max(args.dataset, args.family)
    write_json(
        manifest_path,
        {
            "dataset": args.dataset,
            "family": args.family,
            "title": publication_family_title(args.family),
            "x_max": x_max,
            "plot_path": str(grad_plot_path),
            "plot_paths": {
                "grad_norm_sq": str(grad_plot_path),
                "objective": str(objective_plot_path),
            },
            "entries": manifest_entries,
        },
    )
    try:
        plot_family_from_manifest(manifest_path, y_key="grad_norm_sq")
        plot_family_from_manifest(manifest_path, y_key="objective")
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()
