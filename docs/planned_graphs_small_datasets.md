# plotting contract for small-datasets

## Назначение
Этот файл описывает текущий plotting contract small-datasets ветки в терминах уже
реализованных family presets, CSV-полей и ожидаемых PNG-артефактов.

## Общие правила
- family plots строятся только по CSV через manifest JSON;
- X-axis: `kbits_per_n`;
- publication Y-axis по умолчанию: `grad_norm_sq`;
- обычный run начинается со стартовой CSV-записи `iteration=0`, поэтому графики должны включать точку при `kbits_per_n=0`;
- все кривые внутри одной family figure обрезаются по общему cutoff
  `min(max(kbits_per_n))` между доступными кривыми;
- для `grad_norm_sq` используется логарифмическая шкала по Y.

## Family A
Источник:
- `bits_complexity.experiments.presets.family_a_configs`

Обязательные кривые:
- `EF21 + 32-bit`
- `EF21 + DB(3-bit)`
- `EF21 + DB(4-bit)`
- `EF21 + DB(5-bit)`
- `EF21 + DB(6-bit)`
- `EF21 + Top-10%`
- `EF21 + DB(4-bit) -> Top-10%`

Итоговый файл:
- `outputs/plots/family_A_<dataset>.png`

Manifest:
- `outputs/plots/manifests/<dataset>_family_A.json`

## Family B
Источник:
- `bits_complexity.experiments.presets.family_b_configs`

Обязательные кривые:
- `EF21 + static vs dynamic, biased dynamic dtype, dynamic, p=2, b=4`
- `EF21 + static vs dynamic, biased dynamic dtype, static, p=2, b=4`
- `EF21 + static vs dynamic, biased dynamic dtype, dynamic, p=2, b=3`
- `EF21 + static vs dynamic, biased dynamic dtype, static, p=2, b=3`
- аналогично для `p=4` и `p=8`

Фиксированные параметры:
- `method=ef21`
- `compressor_pipeline=db`
- `quantizer_family=db`
- `bits_per_value in {3, 4}`

Итоговый файл:
- `outputs/plots/family_B_<dataset>_static_vs_dynamic.png`

Manifest:
- `outputs/plots/manifests/<dataset>_family_B.json`

## Family B (DU)
Обязательные кривые:
- тот же набор, что и в Family B, но с `unbiased dynamic dtype`

Фиксированные параметры:
- `method=ef21`
- `compressor_pipeline=db`
- `quantizer_family=du`
- `bits_per_value in {3, 4}`

## Debug plots для single run
Источник:
- `bits_complexity.plots.builder.plot_run_debug_bundle`

Файлы:
- `<metrics_stem>_cum_bits_vs_iteration.png`
- `<metrics_stem>_runtime_vs_kbits_per_n.png`
- `<metrics_stem>_objective_vs_iteration.png`
- `<metrics_stem>_grad_norm_sq_vs_kbits_per_n.png`

Каталог:
- `output_dir/plots/debug/`

## Минимально необходимые CSV-поля
- `iteration`
- `cum_bits`
- `kbits_per_n`
- `objective`
- `grad_norm_sq`
- `runtime_sec`
- поля идентификации запуска: `dataset`, `method`, `compressor_pipeline`, `quantizer_family`, `dynamic_mode`, `p`, `bits_per_value`, `seed`

Семантика стартовой строки:
- обязательна для обычного run;
- `iteration=0`;
- `cum_bits=0`, `kbits_per_n=0`, `step_bits=0`;
- `objective`, `grad_norm_sq`, `accuracy` соответствуют исходной модели до первого шага алгоритма.
