# Проверка `family B` и битовой интерпретации сетки

## Что проверяется

Проверка зафиксирована относительно текущей реализации в `src/bits_complexity/compression/pipelines.py`.

Нужно установить два факта:
- в `family B` режимы `static` и `dynamic` различаются именно правилом обновления общей сетки квантования;
- параметр `b = bits_per_value` либо действительно задаёт алфавит мощности `2^b`, либо в реализации есть расхождение между алфавитом квантователя и моделью битового учёта.

## Канонический набор запусков `family B`

Источник истины для семейства задаётся кодом в `src/bits_complexity/experiments/presets.py`:
- всегда `method=ef21`;
- всегда `compressor_pipeline=db`;
- всегда `quantizer_family=db`;
- всегда `bits_per_value=4`;
- меняются только `dynamic_mode in {static, dynamic}` и `p in {2, 4, 8}`.

Каноническая таблица по committed-артефакту `runs/results/full_parallel/plots/manifests/w8a_family_B.json`:

```text
run_slug;p;dynamic_mode;bits_per_value;config_path;csv_path
w8a__ef21__db__db__dynamic__p2__b4__k0.100__s42;2;dynamic;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p2__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p2__b4__k0.100__s42/metrics.csv
w8a__ef21__db__db__static__p2__b4__k0.100__s42;2;static;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p2__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p2__b4__k0.100__s42/metrics.csv
w8a__ef21__db__db__dynamic__p4__b4__k0.100__s42;4;dynamic;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p4__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p4__b4__k0.100__s42/metrics.csv
w8a__ef21__db__db__static__p4__b4__k0.100__s42;4;static;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p4__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p4__b4__k0.100__s42/metrics.csv
w8a__ef21__db__db__dynamic__p8__b4__k0.100__s42;8;dynamic;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p8__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__dynamic__p8__b4__k0.100__s42/metrics.csv
w8a__ef21__db__db__static__p8__b4__k0.100__s42;8;static;4;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p8__b4__k0.100__s42/config.json;runs/results/full_parallel/runs/w8a/w8a__ef21__db__db__static__p8__b4__k0.100__s42/metrics.csv
```

## Инварианты `static` и `dynamic`

Общая сетка задаётся через `self._current_amax` и `self._levels`.

`static`:
- после первого `prepare_round()` значение `self._current_amax` больше не обновляется;
- следовательно, `_levels = _build_levels(self._current_amax)` остаётся неизменной на всех следующих шагах.

`dynamic`:
- при росте наблюдаемого максимума `observed_amax` сетка расширяется до нового `self._current_amax`;
- при падении масштаба сетка сжимается ступенчато, пока `observed_amax <= self._current_amax / p`;
- значит, `self._levels` меняется только по правилу роста/сжатия с фактором `p`.

Во всех остальных местах `static` и `dynamic` используют одну и ту же квантовку:
- одинаковый `_biased_quantize()` для `EF21 + DB`;
- одинаковый расчёт `bits_for_values`, `bits_for_indices` и `step_bits` в `compress()`.

## Мощность алфавита квантователя

В текущей реализации:
- число положительных уровней по модулю равно `2^(b-1)`;
- знак восстанавливается отдельно через `np.sign(vector)`;
- ноль остаётся отдельным выходным состоянием, потому что `np.sign(0) = 0`.

Поэтому мощность выходного алфавита равна:

```text
1 + 2 * 2^(b - 1) = 2^b + 1
```

Здесь:
- `2^(b-1)` положительных уровней;
- `2^(b-1)` отрицательных уровней;
- `1` отдельный символ нуля.

## Вывод по пункту 2

Бинарный вывод:
- утверждение «текущая реализация кодирует ровно `2^b` различимых состояний на значение» неверно;
- по фактическому выходному алфавиту квантователь даёт `2^b + 1` состояния, если ноль считать отдельным кодируемым символом.

При этом bit accounting в коде всё равно использует контракт:

```text
bits_for_values = transmitted_coordinates * bits_per_value
```

Следовательно, сейчас есть расхождение между:
- фактической мощностью выходного алфавита квантователя;
- моделью битовой стоимости, которая считает ровно `b` бит на переданное значение.
