from __future__ import annotations

import csv
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bits_complexity.common.io import ensure_dir
from bits_complexity.experiments.presets import publication_family_title


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in the active environment."
        ) from exc
    return plt


@dataclass
class CurveData:
    label: str
    x: list[float]
    y: list[float]


FAMILY_FIGSIZE = (8.6, 5.2)
FAMILY_LINEWIDTH = 3.0
FAMILY_MARKERSIZE = 9
FAMILY_MARKER_COUNT = 15
FAMILY_TITLE_FONTSIZE = 16
FAMILY_LABEL_FONTSIZE = 20
FAMILY_TICK_FONTSIZE = 15
FAMILY_LEGEND_FONTSIZE = 16
FAMILY_SAVEFIG_DPI = 300
FAMILY_COLORS = (
    "crimson",
    "forestgreen",
    "darkorange",
    "navy",
    "darkmagenta",
    "darkred",
    "darkgreen",
    "darkcyan",
    "indigo",
    "saddlebrown",
    "teal",
    "maroon",
    "steelblue",
    "darkslateblue",
)
FAMILY_MARKERS = ("o", "s", "*", "^", "h", "d", "p", "8", "X", "P")


def _read_numeric_series(csv_path: Path, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]))
    return xs, ys


_MARKERS = ("o", "*", "v", "^", "<", "s", "D")


def _detect_dtype_label(text: str) -> str | None:
    lowered = text.lower()
    if (
        r"\mathcal{d}_u" in lowered
        or "unbiased dynamic dtype" in lowered
        or "unbiased deynamic dtype" in lowered
        or re.search(r"(^|[^a-z])du([^a-z]|$)", lowered)
    ):
        return r"$\mathcal{D}_U$"
    if (
        r"\mathcal{d}_b" in lowered
        or "biased dynamic dtype" in lowered
        or "biased deynamic dtype" in lowered
        or re.search(r"(^|[^a-z])db([^a-z]|$)", lowered)
    ):
        return r"$\mathcal{D}_B$"
    return None


def _detect_mode_label(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(r"(?:^|,\s*)(dynamic|static)(?:\s*(?:,|$))", lowered)
    if match:
        return match.group(1)
    short_match = re.search(r"(?:^|,\s*)(dyn|st)(?:\s*(?:,|$))", lowered)
    if short_match:
        return "dynamic" if short_match.group(1) == "dyn" else "static"
    return None


def _detect_p_label(text: str) -> str | None:
    explicit = re.search(r"\bp\s*=\s*(\d+)\b", text, flags=re.IGNORECASE)
    if explicit:
        return f"p={explicit.group(1)}"
    compact = re.search(r"\bp(\d+)\b", text, flags=re.IGNORECASE)
    if compact:
        return f"p={compact.group(1)}"
    return None


def _detect_fp_label(text: str) -> str | None:
    fp_match = re.search(r"\bfp\s*([0-9]+)\b", text, flags=re.IGNORECASE)
    if fp_match:
        return f"FP{fp_match.group(1)}"
    bits_match = re.search(r"\bb\s*=\s*([0-9]+)\b", text, flags=re.IGNORECASE)
    if bits_match:
        return f"FP{bits_match.group(1)}"
    compact_bits = re.search(r"\bb([0-9]+)\b", text, flags=re.IGNORECASE)
    if compact_bits:
        return f"FP{compact_bits.group(1)}"
    return None


def _detect_topk_label(text: str) -> str | None:
    match = re.search(r"topk\s*=?\s*([0-9]+(?:\.[0-9]+)?)%", text, flags=re.IGNORECASE)
    if match:
        return f"topk={match.group(1)}%"
    if re.search(r"\btopk\b", text, flags=re.IGNORECASE):
        return "topk"
    return None


def _detect_randk_label(text: str) -> str | None:
    match = re.search(r"randk\s*=?\s*([0-9]+(?:\.[0-9]+)?)%", text, flags=re.IGNORECASE)
    if match:
        return f"randk={match.group(1)}%"
    if re.search(r"\brandk\b", text, flags=re.IGNORECASE):
        return "randk"
    return None


def _compact_curve_label(label: str) -> str:
    normalized = label.strip()
    if not normalized:
        return normalized

    parts: list[str] = []
    dtype = _detect_dtype_label(normalized)
    if dtype is not None:
        parts.append(dtype)
    mode = _detect_mode_label(normalized)
    if mode is not None:
        parts.append(mode)
    p_value = _detect_p_label(normalized)
    if p_value is not None:
        parts.append(p_value)
    fp = _detect_fp_label(normalized)
    if fp is not None:
        parts.append(fp)
    topk = _detect_topk_label(normalized)
    if topk is not None:
        parts.append(topk)
    randk = _detect_randk_label(normalized)
    if randk is not None:
        parts.append(randk)

    if parts:
        ordered_unique = list(dict.fromkeys(parts))
        return ", ".join(ordered_unique)

    replaced = re.sub(r"\bfp\s*([0-9]+)\b", r"FP\1", normalized, flags=re.IGNORECASE)
    replaced = re.sub(r"\bb\s*=\s*([0-9]+)\b", r"FP\1", replaced, flags=re.IGNORECASE)
    replaced = re.sub(r"\bb([0-9]+)\b", r"FP\1", replaced, flags=re.IGNORECASE)
    return replaced


def _normalize_plot_title(dataset: str, title: str) -> str:
    dataset_norm = dataset.strip()
    title_norm = title.strip()
    if not dataset_norm:
        return title_norm
    if not title_norm:
        return dataset_norm
    if dataset_norm.lower() in title_norm.lower():
        return title_norm
    return f"{dataset_norm}: {title_norm}"


def _split_label_parts(label: str) -> list[str]:
    return [part.strip() for part in label.split(",") if part.strip()]


def _sparse_part_kind_and_ratio(part: str) -> tuple[str, str | None] | None:
    normalized = part.strip()
    topk_match = re.fullmatch(
        r"topk\s*=?\s*([0-9]+(?:\.[0-9]+)?)%", normalized, flags=re.IGNORECASE
    )
    if topk_match:
        return "top", topk_match.group(1)
    randk_match = re.fullmatch(
        r"randk\s*=?\s*([0-9]+(?:\.[0-9]+)?)%",
        normalized,
        flags=re.IGNORECASE,
    )
    if randk_match:
        return "rand", randk_match.group(1)
    top_match = re.fullmatch(r"top\s*([0-9]+(?:\.[0-9]+)?)%", normalized, flags=re.IGNORECASE)
    if top_match:
        return "top", top_match.group(1)
    rand_match = re.fullmatch(r"rand\s*([0-9]+(?:\.[0-9]+)?)%", normalized, flags=re.IGNORECASE)
    if rand_match:
        return "rand", rand_match.group(1)
    if re.fullmatch(r"topk", normalized, flags=re.IGNORECASE):
        return "top", None
    if re.fullmatch(r"randk", normalized, flags=re.IGNORECASE):
        return "rand", None
    return None


def _title_contains_part(title: str, part: str) -> bool:
    title_normalized = title.lower()
    if part.lower() in title_normalized:
        return True
    sparse = _sparse_part_kind_and_ratio(part)
    if sparse is None:
        return False
    kind, ratio = sparse
    if ratio is not None:
        ratio_pattern = rf"\b{kind}\s*{re.escape(ratio)}%(?=[^0-9]|$)"
        if re.search(ratio_pattern, title, flags=re.IGNORECASE):
            return True
        explicit_pattern = rf"\b{kind}k\s*=?\s*{re.escape(ratio)}%(?=[^0-9]|$)"
        return re.search(explicit_pattern, title, flags=re.IGNORECASE) is not None
    if re.search(rf"\b{kind}\s*[0-9]+(?:\.[0-9]+)?%(?=[^0-9]|$)", title, flags=re.IGNORECASE):
        return True
    return re.search(rf"\b{kind}k?\b", title, flags=re.IGNORECASE) is not None


def _format_shared_title_part(part: str) -> str:
    sparse = _sparse_part_kind_and_ratio(part)
    if sparse is None:
        return part
    kind, ratio = sparse
    prefix = "Top" if kind == "top" else "Rand"
    if ratio is not None:
        return f"{prefix}{ratio}% compressors"
    return f"{prefix}K compressor"


def _move_shared_legend_parts_to_title(
    curves: list[CurveData],
    title: str,
) -> tuple[list[CurveData], str]:
    if len(curves) < 2:
        return curves, title
    parts_per_curve = [_split_label_parts(curve.label) for curve in curves]
    if not parts_per_curve or any(not parts for parts in parts_per_curve):
        return curves, title

    common_parts: list[str] = []
    other_sets = [set(parts) for parts in parts_per_curve[1:]]
    for part in parts_per_curve[0]:
        if all(part in current for current in other_sets):
            common_parts.append(part)
    if not common_parts:
        return curves, title

    updated_curves: list[CurveData] = []
    for curve, parts in zip(curves, parts_per_curve):
        filtered = [part for part in parts if part not in common_parts]
        if not filtered:
            return curves, title
        updated_curves.append(CurveData(label=", ".join(filtered), x=curve.x, y=curve.y))

    missing_in_title = [
        _format_shared_title_part(part)
        for part in common_parts
        if not _title_contains_part(title, part)
    ]
    missing_in_title = list(dict.fromkeys(missing_in_title))
    if missing_in_title:
        updated_title = f"{title}, {', '.join(missing_in_title)}"
    else:
        updated_title = title
    return updated_curves, updated_title


def _marker_positions(num_points: int, marker_count: int = 10) -> list[int]:
    if num_points <= 0 or marker_count <= 0:
        return []
    if num_points == 1:
        return [0] * marker_count
    if num_points <= marker_count:
        base = list(range(num_points))
        base.extend([num_points - 1] * (marker_count - num_points))
        return base
    step = (num_points - 1) / (marker_count - 1)
    return [round(step * idx) for idx in range(marker_count)]


def _make_style_cycles():
    return itertools.cycle(FAMILY_COLORS), itertools.cycle(FAMILY_MARKERS)


def _unique_by_label(curves: list[CurveData]) -> list[CurveData]:
    seen: dict[str, int] = {}
    resolved: list[CurveData] = []
    for curve in curves:
        count = seen.get(curve.label, 0)
        seen[curve.label] = count + 1
        if count == 0:
            resolved.append(curve)
            continue
        resolved.append(CurveData(label=f"{curve.label} [{count + 1}]", x=curve.x, y=curve.y))
    return resolved


def _clip_curve_to_cutoff(
    x: list[float], y: list[float], cutoff: float, *, eps: float = 1e-12
) -> tuple[list[float], list[float]]:
    if not x or not y:
        return [], []
    trimmed_x: list[float] = []
    trimmed_y: list[float] = []
    for x_i, y_i in zip(x, y):
        if x_i <= cutoff + eps:
            trimmed_x.append(x_i)
            trimmed_y.append(y_i)
    if not trimmed_x:
        return [], []
    if abs(trimmed_x[-1] - cutoff) <= eps:
        trimmed_x[-1] = cutoff
        return trimmed_x, trimmed_y
    if trimmed_x[-1] < cutoff - eps:
        right_idx: int | None = None
        for idx, x_i in enumerate(x):
            if x_i > cutoff + eps:
                right_idx = idx
                break
        if right_idx is not None and right_idx > 0:
            x_left = x[right_idx - 1]
            y_left = y[right_idx - 1]
            x_right = x[right_idx]
            y_right = y[right_idx]
            if x_right - x_left > eps:
                ratio = (cutoff - x_left) / (x_right - x_left)
                y_cutoff = y_left + ratio * (y_right - y_left)
                trimmed_x.append(cutoff)
                trimmed_y.append(y_cutoff)
    return trimmed_x, trimmed_y


def build_family_plot(
    curves: list[CurveData],
    output_path: Path,
    title: str,
    x_label: str = "#Kbits/n",
    y_label: str = r"$\|\nabla f(x)\|^2$",
    log_x: bool = False,
    log_y: bool = True,
    x_max: float | None = None,
    legend_loc: str = "best",
) -> None:
    if not curves:
        raise ValueError("At least one curve is required")
    curves = _unique_by_label(curves)
    common_cutoff = min(max(curve.x) for curve in curves if curve.x)
    if x_max is not None:
        common_cutoff = min(common_cutoff, x_max)
    plt = _load_matplotlib()
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=FAMILY_FIGSIZE)
    ax.set_facecolor("#519dfc")
    ax.patch.set_alpha(0.15)
    color_cycle, marker_cycle = _make_style_cycles()
    for curve in curves:
        xs, ys = _clip_curve_to_cutoff(curve.x, curve.y, common_cutoff)
        if log_x:
            positive_points = [(x_i, y_i) for x_i, y_i in zip(xs, ys) if x_i > 0.0]
            xs = [point[0] for point in positive_points]
            ys = [point[1] for point in positive_points]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=curve.label,
            linewidth=FAMILY_LINEWIDTH,
            color=next(color_cycle),
            marker=next(marker_cycle),
            markersize=FAMILY_MARKERSIZE,
            markevery=_marker_positions(len(xs), marker_count=FAMILY_MARKER_COUNT),
            markeredgecolor="black",
        )
    ax.set_title(title, fontsize=FAMILY_TITLE_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=FAMILY_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FAMILY_LABEL_FONTSIZE)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=FAMILY_TICK_FONTSIZE)
    ax.grid(color="#6d7175", linestyle="--", linewidth=0.7, alpha=0.9)
    legend = ax.legend(
        loc=legend_loc,
        fontsize=FAMILY_LEGEND_FONTSIZE,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.75,
        labelspacing=0.4,
        handlelength=2.5,
        handletextpad=0.8,
        borderpad=0.6,
    )
    legend.get_frame().set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FAMILY_SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_family_from_csvs(
    curve_specs: list[tuple[str, Path]],
    output_path: Path,
    title: str,
    y_key: str = "grad_norm_sq",
    x_max: float | None = None,
) -> None:
    curves = []
    for label, csv_path in curve_specs:
        x, y = _read_numeric_series(csv_path, x_key="kbits_per_n", y_key=y_key)
        curves.append(CurveData(label=_compact_curve_label(label), x=x, y=y))
    curves, title = _move_shared_legend_parts_to_title(curves, title)
    if y_key == "grad_norm_sq":
        y_label = r"$\|\nabla f(x)\|^2$"
    elif y_key == "objective":
        y_label = r"$f(x)$"
    else:
        y_label = y_key
    build_family_plot(
        curves=curves,
        output_path=output_path,
        title=title,
        y_label=y_label,
        log_x=False,
        log_y=(y_key in {"grad_norm_sq", "objective"}),
        x_max=x_max,
    )


def plot_family_from_manifest(manifest_path: Path, y_key: str = "grad_norm_sq") -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    curve_specs = [(entry["label"], Path(entry["csv_path"])) for entry in payload["entries"]]
    plot_paths = payload.get("plot_paths")
    if isinstance(plot_paths, dict):
        selected = plot_paths.get(y_key)
        output_path = Path(str(selected)) if selected is not None else Path(payload["plot_path"])
    else:
        output_path = Path(payload["plot_path"])
    dataset = str(payload["dataset"])
    family = str(payload.get("family") or "")
    canonical_title = publication_family_title(family)
    base_title = canonical_title or str(payload.get("title") or family)
    title = _normalize_plot_title(dataset, base_title)
    x_max = payload.get("x_max")
    plot_family_from_csvs(
        curve_specs=curve_specs,
        output_path=output_path,
        title=title,
        y_key=y_key,
        x_max=float(x_max) if x_max is not None else None,
    )
    return output_path


def plot_run_debug_bundle(csv_path: Path, output_dir: Path) -> None:
    plt = _load_matplotlib()
    ensure_dir(output_dir)
    mappings = [
        (
            "iteration",
            "cum_bits",
            "iteration",
            "cum_bits",
            output_dir / f"{csv_path.stem}_cum_bits_vs_iteration.png",
            False,
        ),
        (
            "kbits_per_n",
            "runtime_sec",
            "#Kbits/n",
            "runtime_sec",
            output_dir / f"{csv_path.stem}_runtime_vs_kbits_per_n.png",
            False,
        ),
        (
            "iteration",
            "objective",
            "iteration",
            "objective",
            output_dir / f"{csv_path.stem}_objective_vs_iteration.png",
            False,
        ),
        (
            "kbits_per_n",
            "grad_norm_sq",
            "#Kbits/n",
            r"$\|\nabla f(x)\|^2$",
            output_dir / f"{csv_path.stem}_grad_norm_sq_vs_kbits_per_n.png",
            True,
        ),
    ]
    for x_key, y_key, x_label, y_label, output_path, log_y in mappings:
        x, y = _read_numeric_series(csv_path, x_key=x_key, y_key=y_key)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
