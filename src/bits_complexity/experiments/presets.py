from __future__ import annotations

from pathlib import Path

from bits_complexity.common.config import RunConfig

SUPPORTED_PUBLICATION_FAMILIES = (
    "db",
    "du",
    "db_topk",
    "du_topk",
    "static_dynamic",
    "static_dynamic_b3",
    "static_dynamic_du",
    "static_dynamic_du_b3",
)

TOPK_LABEL = "topk=10%"


def _dtype_label(quantizer_family: str) -> str:
    if quantizer_family == "db":
        return r"$\mathcal{D}_B$"
    if quantizer_family == "du":
        return r"$\mathcal{D}_U$"
    raise ValueError(f"Unsupported quantizer family: {quantizer_family}")


def _reference_label(*, with_topk: bool) -> str:
    parts = ["FP32"]
    if with_topk:
        parts.append(TOPK_LABEL)
    return ", ".join(parts)


def _quantized_label(quantizer_family: str, bits: int, *, with_topk: bool) -> str:
    parts = [_dtype_label(quantizer_family), f"FP{bits}"]
    if with_topk:
        parts.append(TOPK_LABEL)
    return ", ".join(parts)


def _static_dynamic_label(quantizer_family: str, dynamic_mode: str, p_value: int, bits: int) -> str:
    return ", ".join([_dtype_label(quantizer_family), dynamic_mode, f"p={p_value}", f"FP{bits}"])


def publication_family_title(family: str) -> str | None:
    if family == "db":
        return f"EF21 with {_dtype_label('db')} compressor"
    if family == "du":
        return f"EF21 with {_dtype_label('du')} compressor"
    if family == "db_topk":
        return f"EF21 with {_dtype_label('db')} + Top10% compressors"
    if family == "du_topk":
        return f"EF21 with {_dtype_label('du')} + Top10% compressors"
    if family == "static_dynamic":
        return f"Static vs Dynamic with {_dtype_label('db')} compressor, FP4"
    if family == "static_dynamic_b3":
        return f"Static vs Dynamic with {_dtype_label('db')} compressor, FP3"
    if family == "static_dynamic_du":
        return f"Static vs Dynamic with {_dtype_label('du')} compressor, FP4"
    if family == "static_dynamic_du_b3":
        return f"Static vs Dynamic with {_dtype_label('du')} compressor, FP3"
    return None


def _dense_reference_config(dataset: str, output_dir: Path) -> RunConfig:
    return RunConfig(
        dataset=dataset,
        method="ef21",
        compressor_pipeline="fp32",
        quantizer_family="none",
        dynamic_mode="disabled",
        bits_per_value=32,
        output_dir=output_dir,
    )


def _topk_reference_config(dataset: str, output_dir: Path) -> RunConfig:
    return RunConfig(
        dataset=dataset,
        method="ef21",
        compressor_pipeline="topk",
        quantizer_family="none",
        dynamic_mode="disabled",
        bits_per_value=32,
        output_dir=output_dir,
    )


def _quantized_family_configs(
    dataset: str,
    output_dir: Path,
    *,
    pipeline: str,
    quantizer_family: str,
    reference_config: RunConfig,
    with_topk: bool,
) -> list[tuple[str, RunConfig]]:
    configs: list[tuple[str, RunConfig]] = [
        (_reference_label(with_topk=with_topk), reference_config),
    ]
    for bits in (3, 4, 5, 6):
        configs.append(
            (
                _quantized_label(quantizer_family, bits, with_topk=with_topk),
                RunConfig(
                    dataset=dataset,
                    method="ef21",
                    compressor_pipeline=pipeline,
                    quantizer_family=quantizer_family,
                    dynamic_mode="dynamic",
                    bits_per_value=bits,
                    output_dir=output_dir,
                ),
            )
        )
    return configs


def family_db_configs(dataset: str, output_dir: Path) -> list[tuple[str, RunConfig]]:
    return _quantized_family_configs(
        dataset,
        output_dir,
        pipeline="db",
        quantizer_family="db",
        reference_config=_dense_reference_config(dataset, output_dir),
        with_topk=False,
    )


def family_du_configs(dataset: str, output_dir: Path) -> list[tuple[str, RunConfig]]:
    return _quantized_family_configs(
        dataset,
        output_dir,
        pipeline="db",
        quantizer_family="du",
        reference_config=_dense_reference_config(dataset, output_dir),
        with_topk=False,
    )


def family_db_topk_configs(dataset: str, output_dir: Path) -> list[tuple[str, RunConfig]]:
    return _quantized_family_configs(
        dataset,
        output_dir,
        pipeline="db_topk",
        quantizer_family="db",
        reference_config=_topk_reference_config(dataset, output_dir),
        with_topk=True,
    )


def family_du_topk_configs(dataset: str, output_dir: Path) -> list[tuple[str, RunConfig]]:
    return _quantized_family_configs(
        dataset,
        output_dir,
        pipeline="db_topk",
        quantizer_family="du",
        reference_config=_topk_reference_config(dataset, output_dir),
        with_topk=True,
    )


def _static_dynamic_configs(
    dataset: str,
    output_dir: Path,
    *,
    quantizer_family: str,
    cutoff_kbits_per_n: float,
    bits_per_value: int,
) -> list[tuple[str, RunConfig]]:
    configs: list[tuple[str, RunConfig]] = []
    for p_value in (2, 4, 8):
        cutoff = cutoff_kbits_per_n if bits_per_value == 4 else cutoff_kbits_per_n * 0.75
        for dynamic_mode in ("dynamic", "static"):
            configs.append(
                (
                    _static_dynamic_label(quantizer_family, dynamic_mode, p_value, bits_per_value),
                    RunConfig(
                        dataset=dataset,
                        method="ef21",
                        compressor_pipeline="db",
                        quantizer_family=quantizer_family,
                        dynamic_mode=dynamic_mode,
                        bits_per_value=bits_per_value,
                        p=p_value,
                        cutoff_kbits_per_n=cutoff,
                        output_dir=output_dir,
                    ),
                )
            )
    return configs


def family_static_dynamic_configs(
    dataset: str,
    output_dir: Path,
    *,
    cutoff_kbits_per_n: float = 5.0,
) -> list[tuple[str, RunConfig]]:
    return _static_dynamic_configs(
        dataset,
        output_dir,
        quantizer_family="db",
        cutoff_kbits_per_n=cutoff_kbits_per_n,
        bits_per_value=4,
    )


def family_static_dynamic_b3_configs(
    dataset: str,
    output_dir: Path,
    *,
    cutoff_kbits_per_n: float = 5.0,
) -> list[tuple[str, RunConfig]]:
    return _static_dynamic_configs(
        dataset,
        output_dir,
        quantizer_family="db",
        cutoff_kbits_per_n=cutoff_kbits_per_n,
        bits_per_value=3,
    )


def family_static_dynamic_du_configs(
    dataset: str,
    output_dir: Path,
    *,
    cutoff_kbits_per_n: float = 5.0,
) -> list[tuple[str, RunConfig]]:
    return _static_dynamic_configs(
        dataset,
        output_dir,
        quantizer_family="du",
        cutoff_kbits_per_n=cutoff_kbits_per_n,
        bits_per_value=4,
    )


def family_static_dynamic_du_b3_configs(
    dataset: str,
    output_dir: Path,
    *,
    cutoff_kbits_per_n: float = 5.0,
) -> list[tuple[str, RunConfig]]:
    return _static_dynamic_configs(
        dataset,
        output_dir,
        quantizer_family="du",
        cutoff_kbits_per_n=cutoff_kbits_per_n,
        bits_per_value=3,
    )


def publication_family_configs(
    dataset: str,
    output_dir: Path,
    family: str,
    *,
    cutoff_kbits_per_n: float = 5.0,
) -> list[tuple[str, RunConfig]]:
    if family == "db":
        return family_db_configs(dataset, output_dir)
    if family == "du":
        return family_du_configs(dataset, output_dir)
    if family == "db_topk":
        return family_db_topk_configs(dataset, output_dir)
    if family == "du_topk":
        return family_du_topk_configs(dataset, output_dir)
    if family == "static_dynamic":
        return family_static_dynamic_configs(
            dataset,
            output_dir,
            cutoff_kbits_per_n=cutoff_kbits_per_n,
        )
    if family == "static_dynamic_b3":
        return family_static_dynamic_b3_configs(
            dataset,
            output_dir,
            cutoff_kbits_per_n=cutoff_kbits_per_n,
        )
    if family == "static_dynamic_du":
        return family_static_dynamic_du_configs(
            dataset,
            output_dir,
            cutoff_kbits_per_n=cutoff_kbits_per_n,
        )
    if family == "static_dynamic_du_b3":
        return family_static_dynamic_du_b3_configs(
            dataset,
            output_dir,
            cutoff_kbits_per_n=cutoff_kbits_per_n,
        )
    raise ValueError(f"Unsupported family: {family}")


def family_manifest_path(output_dir: Path, dataset: str, family: str) -> Path:
    return output_dir / "plots" / "manifests" / f"{dataset}_family_{family}.json"


def family_plot_x_max(dataset: str, family: str) -> float | None:
    if dataset != "w8a":
        return None
    if family in {"db_topk", "du_topk"}:
        return 3.0
    if family in {"db", "du"}:
        return 33.0
    if family in {"static_dynamic", "static_dynamic_du"}:
        return 40.0
    return None


def _is_dynamic_vs_static_family(family: str) -> bool:
    return family.startswith("static_dynamic")


def _plot_group_dir_name(*, family: str, y_key: str) -> str:
    if y_key not in {"grad_norm_sq", "objective"}:
        raise ValueError(f"Unsupported y_key: {y_key}")
    if _is_dynamic_vs_static_family(family):
        return (
            "3_dynamic_vs_static_grad_norm"
            if y_key == "grad_norm_sq"
            else "4_dynamic_vs_static_objective"
        )
    return "1_family_grad_norm" if y_key == "grad_norm_sq" else "2_family_objective"


def family_plot_path(
    output_dir: Path,
    dataset: str,
    family: str,
    *,
    y_key: str = "grad_norm_sq",
) -> Path:
    group_dir = output_dir / "plots" / _plot_group_dir_name(family=family, y_key=y_key)
    if family == "db":
        return group_dir / f"ef21_db_{dataset}.png"
    if family == "du":
        return group_dir / f"ef21_du_{dataset}.png"
    if family == "db_topk":
        return group_dir / f"ef21_db_topk_{dataset}.png"
    if family == "du_topk":
        return group_dir / f"ef21_du_topk_{dataset}.png"
    if family == "static_dynamic":
        return group_dir / f"ef21_static_vs_dynamic_{dataset}.png"
    if family == "static_dynamic_b3":
        return group_dir / f"ef21_static_vs_dynamic_b3_{dataset}.png"
    if family == "static_dynamic_du":
        return group_dir / f"ef21_static_vs_dynamic_du_{dataset}.png"
    if family == "static_dynamic_du_b3":
        return group_dir / f"ef21_static_vs_dynamic_du_b3_{dataset}.png"
    raise ValueError(f"Unsupported family: {family}")
