# Small-Datasets Runbook

## Назначение
Этот файл фиксирует operational path для текущего проекта: какие проверки доступны,
какие артефакты считаются обязательными и что проверять после короткого smoke-run.
Полное описание интерфейсов и defaults находится в
[`INTERFACES_AND_TERMS.md`](./INTERFACES_AND_TERMS.md).

## Базовые команды
- single run:
  - `uv run --group dev python -m bits_complexity.experiments.run ...`
- family orchestration:
  - `uv run --group dev python -m bits_complexity.experiments.run_family ...`
- publication plot:
  - `uv run --group dev python -m bits_complexity.plots.run_publication ...`
- quality gate:
  - `uv run --group dev python -m bits_complexity.dev.quality_gate all`
- smoke pipeline:
  - `uv run --group dev python -m bits_complexity.dev.smoke_pipeline ...`

## Runtime и dev-зависимости
- runtime: `numpy`
- для plotting нужен `matplotlib`
- для lint нужен `ruff`
- для coverage gate нужен модуль `trace` из stdlib
- тестовый стек: `unittest`

## Expected outputs
- single run:
  - `output_dir/runs/<dataset>/<run_slug>/metrics.csv`
  - `output_dir/runs/<dataset>/<run_slug>/config.json`
- family run:
  - single-run CSV/config для всех кривых семьи
  - `output_dir/plots/manifests/<dataset>_family_<family>.json`
  - итоговый family PNG, если установлен `matplotlib`
- smoke pipeline:
  - `output_dir/smoke_summary.json`
  - один `EF21 + DB(4-bit) -> Top-10%` run на `a9a`
  - один `DIANA + DU` run на `a9a`
  - один smoke manifest `a9a_family_A_smoke.json`

## Smoke-check checklist
После `uv run --group dev python -m bits_complexity.dev.smoke_pipeline --force` проверить:
- существует `smoke_summary.json`
- пути из `single_run`, `diana_run`, `family_run` существуют на диске
- `diana_run` использует `quantizer_family = du` в `config.json`
- smoke manifest содержит `entries`
- первая строка обычного `metrics.csv` имеет `iteration=0` и нулевой bit budget
- в CSV поле `cum_bits` монотонно не убывает
- `kbits_per_n` доходит до short cutoff или run заканчивается по `max_iterations`

## Validation path
Проверка test suite:

```bash
uv run --group dev python -m unittest discover -s tests -v
```

Проверка coverage gate:

```bash
uv run --group dev python -m bits_complexity.dev.quality_gate coverage
```

Проверка полного quality gate:

```bash
uv run --group dev python -m bits_complexity.dev.quality_gate all
```

## Troubleshooting
- `matplotlib is required for plotting`: установите `matplotlib` или запускайте только runner/manifest generation.
- `ruff is not installed`: установите `ruff` или используйте `quality_gate all --skip-lint`.
- `Coverage gate failed`: coverage считается только по `src/bits_complexity`, исключая `src/bits_complexity/dev`.
- `EF21 must use biased DB semantics, not DU`: для finite-grid ветки используйте `quantizer_family=db`.
- `DIANA must use unbiased DU semantics, not DB`: для finite-grid ветки используйте `quantizer_family=du`.

## Служебные замечания
- `--force` нужен для перезаписи уже существующих артефактов запуска.
- без `--force` runner переиспользует run, только если одновременно существуют и `metrics.csv`, и `config.json`.
- `--dry-run` у single run создает `config.json` и пустой `metrics.csv` без исполнения метода.
- plotting для family выполняется best-effort: manifest создается даже без доступного backend.
