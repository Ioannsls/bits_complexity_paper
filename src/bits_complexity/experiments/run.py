from __future__ import annotations

import argparse
from pathlib import Path

from bits_complexity.common.config import (
    SUPPORTED_DATASETS,
    SUPPORTED_DYNAMIC_MODES,
    SUPPORTED_METHODS,
    SUPPORTED_PIPELINES,
    SUPPORTED_QUANTIZER_FAMILIES,
    RunConfig,
)
from bits_complexity.experiments.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Запустить один small-datasets эксперимент с явной конфигурацией метода, "
            "компрессора и bit-accounting."
        ),
        epilog=(
            "Finite-grid пайплайны поддерживают как DB, так и DU семантику через "
            "--quantizer-family. Для DIANA запрещён DB-режим."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Датасет LibSVM-формата. Поддерживаются: %(choices)s.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=SUPPORTED_METHODS,
        help="Оптимизационный метод. Поддерживаются: %(choices)s.",
    )
    parser.add_argument(
        "--compressor-pipeline",
        required=True,
        choices=SUPPORTED_PIPELINES,
        help="Пайплайн компрессии: fp32, finite-grid, sparsification или их композиция.",
    )
    parser.add_argument(
        "--quantizer-family",
        default="none",
        choices=SUPPORTED_QUANTIZER_FAMILIES,
        help="Семейство finite-grid квантизатора. 'none' включает автоматическое разрешение.",
    )
    parser.add_argument(
        "--dynamic-mode",
        default="disabled",
        choices=SUPPORTED_DYNAMIC_MODES,
        help="Режим обновления scale/grid для finite-grid компрессии.",
    )
    parser.add_argument(
        "--bits-per-value",
        type=int,
        default=4,
        help="Число бит на одно значение в finite-grid ветке. По умолчанию: %(default)s.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=2,
        help="База геометрической rounding lattice в finite-grid компрессии.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Количество клиентов после разбиения train-выборки. По умолчанию: %(default)s.",
    )
    parser.add_argument(
        "--cutoff-kbits-per-n",
        type=float,
        default=5.0,
        help="Коммуникационный бюджет остановки в единицах kbits / n.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Случайное зерно для разбиения данных и стохастической компрессии.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Корневой каталог, куда будут записаны CSV, config и графики.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Максимальное число итераций метода.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Шаг оптимизации.",
    )
    parser.add_argument(
        "--auto-learning-rate",
        action="store_true",
        help="Автоматически подобрать максимальный lr с монотонным убыванием grad_norm_sq.",
    )
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Множитель, домножаемый на итоговый effective learning rate.",
    )
    parser.add_argument(
        "--diana-alpha",
        type=float,
        default=1.0,
        help="Коэффициент обновления shift-переменной в DIANA.",
    )
    parser.add_argument(
        "--k-ratio",
        type=float,
        default=0.1,
        help="Доля координат для TopK/RandK-пайплайнов.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Каталог с файлами датасетов.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Построить debug-графики после завершения запуска.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Создать только config и пустой CSV без исполнения метода.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие артефакты запуска с тем же slug.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RunConfig(
        dataset=args.dataset,
        method=args.method,
        compressor_pipeline=args.compressor_pipeline,
        quantizer_family=args.quantizer_family,
        dynamic_mode=args.dynamic_mode,
        bits_per_value=args.bits_per_value,
        p=args.p,
        clients=args.clients,
        cutoff_kbits_per_n=args.cutoff_kbits_per_n,
        seed=args.seed,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        learning_rate_auto=args.auto_learning_rate,
        learning_rate_multiplier=args.learning_rate_multiplier,
        diana_alpha=args.diana_alpha,
        k_ratio=args.k_ratio,
        plot=args.plot,
        dry_run=args.dry_run,
        force=args.force,
    )
    artifacts = run_experiment(config=config, datasets_dir=args.datasets_dir)
    print(artifacts.csv_path)


if __name__ == "__main__":
    main()
