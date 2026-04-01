# Datasets policy

Каталог `datasets/` используется только как локальный runtime-вход для CLI.
Сами датасеты LibSVM (`*.txt`) в git не коммитятся (policy: `code-only`).

## Поддерживаемые имена

- `mushrooms.txt`
- `a9a.txt`
- `a9a_test.txt`
- `w8a.txt`
- `w8a_test.txt`

## Как подготовить локально

1. Скачайте датасеты из официальных источников LibSVM.
2. Положите файлы в этот каталог с именами из списка выше.
3. Проверьте запуск:

```bash
uv run --group dev python -m bits_complexity.experiments.run \
  --dataset a9a \
  --method ef21 \
  --compressor-pipeline fp32 \
  --output-dir outputs
```

Если файлов нет, загрузчик данных завершится понятной ошибкой `FileNotFoundError`.
