from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CLIENTS = 10
DEFAULT_BITS_PER_VALUE = 4
DEFAULT_SEED = 42
DEFAULT_K_RATIO = 0.1
DEFAULT_CUTOFF_KBITS_PER_N = 5.0
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_L2_REG = 0.001

SUPPORTED_DATASETS = ("mushrooms", "a9a", "w8a")
SUPPORTED_METHODS = ("ef21", "diana")
SUPPORTED_PIPELINES = ("fp32", "db", "topk", "randk", "db_topk")
SUPPORTED_DYNAMIC_MODES = ("disabled", "static", "dynamic")
SUPPORTED_QUANTIZER_FAMILIES = ("none", "db", "du")


@dataclass
class RunConfig:
    """Конфигурация одного small-datasets запуска.

    Notes
    -----
    * `p` — база геометрической rounding lattice, а не порядок нормы.
    * `kbits_per_n` в метриках — накопленные килобиты, делённые на число признаков `n`.
    * Для logistic-regression paper preset по умолчанию используется `l2_reg=0.001`.
    """

    dataset: str
    method: str
    compressor_pipeline: str
    quantizer_family: str = "none"
    dynamic_mode: str = "disabled"
    bits_per_value: int = DEFAULT_BITS_PER_VALUE
    p: int = 2
    clients: int = DEFAULT_CLIENTS
    cutoff_kbits_per_n: float = DEFAULT_CUTOFF_KBITS_PER_N
    seed: int = DEFAULT_SEED
    output_dir: Path = Path("outputs")
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    learning_rate: float = 0.01
    learning_rate_auto: bool = False
    learning_rate_multiplier: float = 1.0
    diana_alpha: float = 1.0
    k_ratio: float = DEFAULT_K_RATIO
    l2_reg: float = DEFAULT_L2_REG
    plot: bool = False
    dry_run: bool = False
    force: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data

    @property
    def run_slug(self) -> str:
        return self.run_slug_with_learning_rate()

    def run_slug_with_learning_rate(self, resolved_learning_rate: float | None = None) -> str:
        lr_token = self.learning_rate_slug_token(resolved_learning_rate=resolved_learning_rate)
        return (
            f"{self.dataset}__{self.method}__{self.compressor_pipeline}"
            f"__{self.resolved_quantizer_family}"
            f"__{self.dynamic_mode}__p{self.p}__b{self.bits_per_value}"
            f"__k{self.k_ratio:.3f}__{lr_token}__s{self.seed}"
        )

    @property
    def resolved_quantizer_family(self) -> str:
        if self.quantizer_family != "none":
            return self.quantizer_family
        if self.compressor_pipeline in {"db", "db_topk"}:
            if self.method == "ef21":
                return "db"
            if self.method == "diana":
                return "du"
        return "none"

    def learning_rate_slug_token(self, resolved_learning_rate: float | None = None) -> str:
        effective_learning_rate = (
            self.learning_rate if resolved_learning_rate is None else resolved_learning_rate
        )
        prefix = "lra" if self.learning_rate_auto else "lr"
        return f"{prefix}{effective_learning_rate:.12g}"


def validate_config(config: RunConfig) -> None:
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if config.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {config.method}")
    if config.compressor_pipeline not in SUPPORTED_PIPELINES:
        raise ValueError(f"Unsupported pipeline: {config.compressor_pipeline}")
    if config.dynamic_mode not in SUPPORTED_DYNAMIC_MODES:
        raise ValueError(f"Unsupported dynamic mode: {config.dynamic_mode}")
    if config.quantizer_family not in SUPPORTED_QUANTIZER_FAMILIES:
        raise ValueError(f"Unsupported quantizer family: {config.quantizer_family}")
    if config.clients <= 0:
        raise ValueError("clients must be positive")
    if config.bits_per_value <= 0:
        raise ValueError("bits_per_value must be positive")
    if config.p <= 1:
        raise ValueError("p must be greater than 1")
    if not 0 < config.k_ratio <= 1:
        raise ValueError("k_ratio must be in (0, 1]")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.learning_rate_multiplier <= 0:
        raise ValueError("learning_rate_multiplier must be positive")
    if config.cutoff_kbits_per_n <= 0:
        raise ValueError("cutoff_kbits_per_n must be positive")
    if config.l2_reg < 0:
        raise ValueError("l2_reg must be non-negative")
    if config.method == "diana" and config.resolved_quantizer_family == "db":
        raise ValueError("DIANA must use unbiased DU semantics, not DB")
