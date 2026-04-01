# Интерфейсы и термины small-datasets ветки

## Назначение
Этот файл является каноническим reference для текущих CLI-интерфейсов, `RunConfig`,
family presets, CSV/manifest contracts и plotting behavior в small-datasets ветке.
Если markdown расходится с кодом, источником истины считается код в `src/bits_complexity`.

## Термины
- `method`: оптимизационный метод. Поддерживаются `ef21`, `diana`.
- `compressor_pipeline`: схема компрессии градиента. Поддерживаются `fp32`, `db`, `topk`, `randk`, `db_topk`.
- `quantizer_family`: finite-grid семейство. Поддерживаются `none`, `db`, `du`.
- `dynamic_mode`: режим обновления finite-grid сетки. Поддерживаются `disabled`, `static`, `dynamic`.
- `family A/B`: предопределенные наборы запусков для publication plots.
- `single run`: один запуск с конкретным `RunConfig`.
- `publication plot`: график по manifest JSON без повторного запуска экспериментов.

## CLI

### `bits_complexity.experiments.run`
Команда:

```bash
PYTHONPATH=src python3 -m bits_complexity.experiments.run ...
```

Назначение:
- запускает один эксперимент;
- пишет `metrics.csv` и `config.json`;
- при `--plot` дополнительно строит debug bundle.

Аргументы:
- `--dataset`, `str`, обязательно, choices: `mushrooms`, `a9a`, `w8a`.
- `--method`, `str`, обязательно, choices: `ef21`, `diana`.
- `--compressor-pipeline`, `str`, обязательно, choices: `fp32`, `db`, `topk`, `randk`, `db_topk`.
- `--quantizer-family`, `str`, default: `none`, choices: `none`, `db`, `du`.
- `--dynamic-mode`, `str`, default: `disabled`, choices: `disabled`, `static`, `dynamic`.
- `--bits-per-value`, `int`, default: `4`.
- `--p`, `int`, default: `2`.
- `--clients`, `int`, default: `10`.
- `--cutoff-kbits-per-n`, `float`, default: `5.0`.
- `--seed`, `int`, default: `42`.
- `--output-dir`, `Path`, default: `outputs`.
- `--max-iterations`, `int`, default: `200`.
- `--learning-rate`, `float`, default: `0.01`.
- `--diana-alpha`, `float`, default: `1.0`.
- `--k-ratio`, `float`, default: `0.1`.
- `--datasets-dir`, `Path`, default: `datasets`.
- `--plot`, flag, default: `False`.
- `--dry-run`, flag, default: `False`.
- `--force`, flag, default: `False`.

Выход:
- `output_dir/runs/<dataset>/<run_slug>/metrics.csv`
- `output_dir/runs/<dataset>/<run_slug>/config.json`
- при `--plot`: PNG-файлы в `output_dir/plots/debug/`

Поведение:
- конфиг валидируется до запуска;
- если `metrics.csv` и `config.json` уже существуют и `--force` не задан, артефакты переиспользуются;
- `--dry-run` пишет `config.json` и пустой `metrics.csv` без запуска метода;
- `--plot` строит debug bundle после записи CSV.

### `bits_complexity.experiments.run_family`
Команда:

```bash
PYTHONPATH=src python3 -m bits_complexity.experiments.run_family ...
```

Назначение:
- запускает предопределенное семейство конфигураций;
- пишет single-run артефакты для каждой кривой;
- собирает manifest JSON для plotting;
- пытается сразу построить publication plot.

Аргументы:
- `--dataset`, `str`, обязательно.
- `--family`, `str`, обязательно, choices: `A`, `B`.
- `--datasets-dir`, `Path`, default: `datasets`.
- `--output-dir`, `Path`, default: `outputs`.
- `--max-iterations`, `int`, default: `200`.
- `--cutoff-kbits-per-n`, `float`, default: `5.0`.
- `--seed`, `int`, default: `42`.
- `--force`, flag, default: `False`.

Family presets:
- `A`:
  - `EF21 + 32-bit`
  - `EF21 + DB(3-bit)` with `dynamic_mode=dynamic`
  - `EF21 + DB(4-bit)` with `dynamic_mode=dynamic`
  - `EF21 + DB(5-bit)` with `dynamic_mode=dynamic`
  - `EF21 + DB(6-bit)` with `dynamic_mode=dynamic`
  - `EF21 + Top-10%`
  - `EF21 + DB(4-bit) -> Top-10%` with `dynamic_mode=dynamic`
- `B`:
  - `Dynamic, p=2`
  - `Static, p=2`
  - `Dynamic, p=4`
  - `Static, p=4`
  - `Dynamic, p=8`
  - `Static, p=8`
  - все кривые `family B` используют `method=ef21`, `compressor_pipeline=db`, `quantizer_family=db`, `bits_per_value=4`

Выход:
- single-run артефакты для всех кривых семьи;
- `output_dir/plots/manifests/<dataset>_family_<family>.json`;
- `output_dir/plots/family_A_<dataset>.png` или `output_dir/plots/family_B_<dataset>_static_vs_dynamic.png`, если доступен `matplotlib`.

Поведение:
- `run_family` не требует отдельного шага manifest generation;
- если plotting backend недоступен, manifest все равно будет создан.

### `bits_complexity.plots.run_publication`
Команда:

```bash
PYTHONPATH=src python3 -m bits_complexity.plots.run_publication ...
```

Назначение:
- строит publication plot по существующему manifest JSON;
- не перезапускает эксперименты.

Аргументы:
- `--dataset`, `str`, обязательно.
- `--family`, `str`, обязательно, choices: `A`, `B`.
- `--output-dir`, `Path`, default: `outputs/plots`.
- `--manifest`, `Path`, default: `None`.

Поведение:
- если `--manifest` не задан, используется стандартный путь через `family_manifest_path(args.output_dir.parent, dataset, family)`;
- путь к итоговому PNG берется из содержимого manifest JSON.

### `bits_complexity.dev.quality_gate`
Команда:

```bash
PYTHONPATH=src python3 -m bits_complexity.dev.quality_gate ...
```

Назначение:
- запускает `ruff`, `unittest`, coverage gate или полный набор проверок.

Аргументы:
- `command`, обязательно, choices: `lint`, `test`, `coverage`, `all`.
- `--coverage-threshold`, `float`, default: `90.0`.
- `--skip-lint`, flag, default: `False`.

Поведение:
- `lint` запускает `ruff check` и `ruff format --check`;
- `test` запускает `python -m unittest discover -s tests -v`;
- `coverage` использует `python -m trace --count --summary` и считает покрытие только по `src/bits_complexity`, исключая `src/bits_complexity/dev`;
- `all` последовательно запускает lint, tests и coverage.

### `bits_complexity.dev.smoke_pipeline`
Команда:

```bash
PYTHONPATH=src python3 -m bits_complexity.dev.smoke_pipeline ...
```

Назначение:
- выполняет короткую end-to-end проверку рабочего пути на реальных данных.

Аргументы:
- `--datasets-dir`, `Path`, default: `datasets`.
- `--output-dir`, `Path`, default: `outputs_smoke_final`.
- `--seed`, `int`, default: `42`.
- `--max-iterations`, `int`, default: `3`.
- `--cutoff-kbits-per-n`, `float`, default: `0.05`.
- `--force`, flag, default: `False`.

Что запускает:
- один `EF21 + DB(4-bit) -> Top-10%` на `a9a` c `dynamic_mode=static`;
- один `DIANA + DU` на `a9a`;
- короткий smoke manifest с двумя A-like кривыми: `EF21 + 32-bit` и `EF21 + Top-10%`.

Выход:
- `output_dir/smoke_summary.json`;
- связанные single-run CSV/config артефакты;
- `output_dir/plots/manifests/a9a_family_A_smoke.json`.

## Python API

### `RunConfig`
Модуль: `bits_complexity.common.config`

Назначение:
- единая конфигурация single run;
- используется CLI, presets и раннером.

Поля:
- обязательные: `dataset`, `method`, `compressor_pipeline`.
- finite-grid и режимы: `quantizer_family`, `dynamic_mode`, `bits_per_value`, `p`.
- execution: `clients`, `cutoff_kbits_per_n`, `seed`, `output_dir`, `max_iterations`, `plot`, `dry_run`, `force`.
- optimization: `learning_rate`, `diana_alpha`, `k_ratio`, `l2_reg`.

Defaults:
- `quantizer_family="none"`
- `dynamic_mode="disabled"`
- `bits_per_value=4`
- `p=2`
- `clients=10`
- `cutoff_kbits_per_n=5.0`
- `seed=42`
- `output_dir=Path("outputs")`
- `max_iterations=200`
- `learning_rate=0.01`
- `diana_alpha=1.0`
- `k_ratio=0.1`
- `l2_reg=0.001`
- `plot=False`
- `dry_run=False`
- `force=False`

Вычисляемые свойства:
- `run_slug`: стабильное имя каталога запуска
  `dataset__method__compressor_pipeline__resolved_quantizer_family__dynamic_mode__p{p}__b{bits}__k{k_ratio}__s{seed}`.
- `resolved_quantizer_family`:
  - возвращает `quantizer_family`, если оно не равно `none`;
  - для `db` и `db_topk` выводит `db` при `method=ef21` и `du` при `method=diana`;
  - для остальных пайплайнов возвращает `none`.

### `validate_config(config)`
Проверяет:
- допустимые dataset/method/pipeline/dynamic mode/quantizer family;
- положительность `clients`, `bits_per_value`, `max_iterations`, `cutoff_kbits_per_n`;
- ограничение `p > 1`;
- ограничение `0 < k_ratio <= 1`;
- ограничение `l2_reg >= 0`;
- запрет `EF21 + DU`;
- запрет `DIANA + DB`.

### `run_experiment(config, datasets_dir)`
Модуль: `bits_complexity.experiments.runner`

Вход:
- `config: RunConfig`
- `datasets_dir: Path`

Выход:
- `RunArtifacts(csv_path, config_path, reused_existing)`

Поведение:
- валидирует конфиг;
- загружает датасет и разбивает train-часть на клиентов;
- строит `LogisticProblem`, компрессорный pipeline и training method;
- пишет CSV и `config.json`;
- для обычного запуска добавляет в начало CSV стартовую запись с состоянием до первого шага и до method-specific обменов;
- останавливает run при достижении `cutoff_kbits_per_n` или `max_iterations`;
- при `dry_run=True` пишет только config и пустой CSV.

`config.json` дополнительно содержит:
- `feature_dim`
- `train_size`
- `dynamic_rule`
- `records` для обычного запуска, равное числу строк в `metrics.csv` включая стартовую запись, или `dry_run=True` для dry-run.

## CSV contract
CSV-поля задаются `bits_complexity.methods.base.CSV_FIELDS`:
- `seed`
- `dataset`
- `method`
- `compressor_pipeline`
- `quantizer_family`
- `dynamic_mode`
- `p`
- `bits_per_value`
- `k`
- `k_ratio`
- `iteration`
- `transmitted_coordinates`
- `bits_for_values`
- `bits_for_indices`
- `step_bits`
- `cum_bits`
- `kbits_per_n`
- `objective`
- `grad_norm_sq`
- `accuracy`
- `runtime_sec`

Семантика:
- publication-oriented plotting по умолчанию использует `kbits_per_n` по оси X и `grad_norm_sq` по оси Y;
- `objective` и `accuracy` сохраняются как дополнительные диагностические метрики;
- `k` вычисляется как `max(1, round(k_ratio * feature_dim))`;
- первая строка обычного run имеет `iteration=0`, `transmitted_coordinates=0`, `bits_for_values=0`, `bits_for_indices=0`, `step_bits=0`, `cum_bits=0`, `kbits_per_n=0`;
- метрики в строке `iteration=0` вычисляются на `problem.initial_point()` до запуска алгоритма.

## Manifest contract
Manifest JSON содержит:
- `dataset`
- `family`
- `plot_path`
- `entries`

Каждый элемент `entries` содержит:
- `label`
- `csv_path`
- `config_path`
- `dataset`
- `family`
- `method`
- `compressor_pipeline`
- `quantizer_family`
- `dynamic_mode`
- `bits_per_value`
- `p`
- `seed`

## Plotting behavior
- `plot_family_from_manifest` читает CSV из `entries[*].csv_path` и пишет PNG в `plot_path`;
- family plots строятся по `kbits_per_n` против `grad_norm_sq`, если явно не задан другой `y_key`;
- перед отрисовкой все кривые обрезаются до общего cutoff, равного `min(max(x))` между кривыми с непустыми данными;
- если в CSV присутствует стартовая запись `iteration=0`, графики начинаются с точки `kbits_per_n=0`;
- Y-axis логарифмический только для `grad_norm_sq`.

Debug bundle для одного запуска строит:
- `cum_bits` vs `iteration`
- `runtime_sec` vs `kbits_per_n`
- `objective` vs `iteration`
- `grad_norm_sq` vs `kbits_per_n`

## Инварианты
- `EF21` использует biased finite-grid семантику `DB`.
- `DIANA` использует unbiased finite-grid семантику `DU`.
- publication plots строятся только по CSV/manifest артефактам.
- reuse существующих артефактов допускается только если одновременно существуют и `metrics.csv`, и `config.json`, и не задан `force`.
