from __future__ import annotations

import argparse
from pathlib import Path

from bits_complexity.experiments.presets import SUPPORTED_PUBLICATION_FAMILIES, family_manifest_path
from bits_complexity.plots.builder import plot_family_from_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Построить publication plot по manifest-driven описанию семейства CSV-кривых."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Имя датасета, используемое для стандартного пути manifest.",
    )
    parser.add_argument(
        "--family",
        choices=SUPPORTED_PUBLICATION_FAMILIES,
        required=True,
        help="Идентификатор семейства графиков.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "plots",
        help="Каталог plots; используется для вывода стандартного manifest path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Явный путь к manifest JSON. Если не задан, используется стандартный путь.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest or family_manifest_path(
        args.output_dir.parent, args.dataset, args.family
    )
    output_path = plot_family_from_manifest(manifest_path)
    print(output_path)


if __name__ == "__main__":
    main()
