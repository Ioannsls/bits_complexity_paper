# bits-complexity-small-datasets

Small-datasets библиотека и CLI для экспериментов по communication-aware compression
(`mushrooms`, `a9a`, `w8a`) с logistic regression, явным bit accounting и
manifest-driven plotting.

## Quickstart (канонический)

### 1) Установка

```bash
uv sync --group dev
```

### 2) Один запуск

```bash
uv run --group dev python -m bits_complexity.experiments.run \
  --dataset a9a \
  --method ef21 \
  --compressor-pipeline db_topk \
  --quantizer-family db \
  --dynamic-mode dynamic \
  --bits-per-value 4 \
  --p 2 \
  --output-dir outputs
```

### 3) Family запуск

```bash
uv run --group dev python -m bits_complexity.experiments.run_family \
  --dataset a9a \
  --family db \
  --output-dir outputs
```

### 4) Проверки качества

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev python -m unittest discover -s tests -v
```

### 5) Сборка пакета

```bash
uv build
uvx twine check dist/*
```

## Данные и артефакты

Репозиторий следует политике `code-only`: большие датасеты и результаты запусков
(`runs/`, `outputs/`) в git не хранятся.

- Инструкции по датасетам: [`datasets/README.md`](datasets/README.md)
- Контракты и интерфейсы: [`docs/INTERFACES_AND_TERMS.md`](docs/INTERFACES_AND_TERMS.md)
- Runbook: [`docs/RUNBOOK_small_datasets.md`](docs/RUNBOOK_small_datasets.md)
