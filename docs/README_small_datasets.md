# Small-Datasets Project

## Назначение
Этот репозиторий содержит small-datasets ветку для экспериментов на `mushrooms`, `a9a`, `w8a`
с logistic regression, явным bit accounting и manifest-driven plotting.

Основной код находится в `src/bits_complexity`. Канонический reference по интерфейсам и
контрактам расположен в [`INTERFACES_AND_TERMS.md`](./INTERFACES_AND_TERMS.md).

## Установка
Рекомендуемое окружение:

```bash
uv sync --group dev
```

## Структура
- `src/bits_complexity/data` — загрузка LibSVM-датасетов и split на клиентов.
- `src/bits_complexity/problems` — logistic objective и метрики.
- `src/bits_complexity/compression` — компрессоры и pipeline composition.
- `src/bits_complexity/methods` — `EF21`, `DIANA`, CSV contract.
- `src/bits_complexity/experiments` — single run, family orchestration, artifact writing.
- `src/bits_complexity/plots` — publication/debug plotting по CSV и manifest.
- `src/bits_complexity/dev` — quality gate и smoke pipeline.
- `tests` — `unittest` suite.
- `docs` — пользовательская, reference и operational документация.

## Быстрый старт
Канонический quickstart вынесен в корневой [`README.md`](../README.md), чтобы не дублировать
команды запуска в нескольких местах.

## Что появится на диске
- `outputs/runs/<dataset>/<run_slug>/metrics.csv`
- `outputs/runs/<dataset>/<run_slug>/config.json`
- `outputs/plots/manifests/<dataset>_family_<family>.json`
- `outputs/plots/family_A_<dataset>.png` или `outputs/plots/family_B_<dataset>_static_vs_dynamic.png`
- при `--plot`: debug PNG в `outputs/plots/debug/`

## Проверки
Тесты:

```bash
uv run --group dev python -m unittest discover -s tests -v
```

Quality gate:

```bash
uv run --group dev python -m bits_complexity.dev.quality_gate all
```

Smoke pipeline:

```bash
uv run --group dev python -m bits_complexity.dev.smoke_pipeline \
  --datasets-dir datasets \
  --output-dir outputs_smoke_final \
  --force
```

## Дополнительные документы
- [`INTERFACES_AND_TERMS.md`](./INTERFACES_AND_TERMS.md) — канонический reference по CLI, `RunConfig`, CSV/manifest contracts и plotting behavior.
- [`RUNBOOK_small_datasets.md`](./RUNBOOK_small_datasets.md) — operational checklist, troubleshooting и smoke-проверки.
- [`planned_graphs_small_datasets.md`](./planned_graphs_small_datasets.md) — текущий plotting contract для family A/B и debug-графиков.
