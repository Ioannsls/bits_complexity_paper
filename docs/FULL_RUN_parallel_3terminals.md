# Full Run in One Launcher

Этот файл фиксирует единый launcher для полного small-datasets прогона с ограничением параллелизма.

## Что покрывается
- `db` family для `mushrooms`, `a9a`, `w8a`
- `du` family для `mushrooms`, `a9a`, `w8a`
- `db_topk` family для `mushrooms`, `a9a`, `w8a`
- `du_topk` family для `mushrooms`, `a9a`, `w8a`
- `static_dynamic` family для `mushrooms`, `a9a`, `w8a`
- `static_dynamic_du` family для `mushrooms`, `a9a`, `w8a`

Итого:
- `12` графиков с reference + 3/4/5/6-bit finite-grid кривыми
- `6` графиков static vs dynamic
- `18` manifest-driven plot суммарно

## Фиксированные параметры
- `datasets-dir = datasets`
- `output-dir = runs/results/full_parallel`
- базовый `cutoff-kbits-per-n = 200`
- для `db_topk` и `du_topk`: `100`
- для `w8a` в `db_topk` и `du_topk`: `10`
- для `w8a` в `static_dynamic` и `static_dynamic_du`: `70`
- `max-iterations = 1000000`
- `seed = 42`
- `force = true`

## Планировщик задач
- используется один Python CLI `python -m bits_complexity.experiments.run_parallel --config runs/configs/full_run_parallel.json`
- все `5` family для каждого датасета запускаются из одного очередного launcher
- одновременно выполняется не более `6` задач
- для всех family-run включён `--auto-learning-rate`

## Как запускать
Из корня репозитория:

```bash
python -m bits_complexity.experiments.run_parallel --config runs/configs/full_run_parallel.json
```

Для варианта `n=100`:

```bash
python -m bits_complexity.experiments.run_parallel --config runs/configs/full_run_parallel_n100.json
```

## Ожидаемые артефакты
- `runs/results/full_parallel/runs/<dataset>/<run_slug>/metrics.csv`
- `runs/results/full_parallel/runs/<dataset>/<run_slug>/config.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_db.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_du.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_db_topk.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_du_topk.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_static_dynamic.json`
- `runs/results/full_parallel/plots/manifests/<dataset>_family_static_dynamic_du.json`
- family PNG в `runs/results/full_parallel/plots/`

## Проверка покрытия
- все `5` manifest'ов должны появиться для `mushrooms`, `a9a`, `w8a`
- все `6` manifest'ов должны появиться для `mushrooms`, `a9a`, `w8a`
- для `db/du/db_topk/du_topk` должно быть по `5` кривых на manifest
- для `static_dynamic` и `static_dynamic_du` должно быть по `12` кривых на manifest
- остановка должна происходить по budget правилам family/dataset, а не по sentinel `max-iterations`
