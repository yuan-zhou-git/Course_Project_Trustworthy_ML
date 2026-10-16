# Results

Runs: {'base': 'base_v1', 'random': 'random15k_v1', 'advanced': 'advanced15k_v1'}

## Per-benchmark scores (pass@8 / pass@1)

| benchmark | n | base pass@8 | base pass@1 | random pass@8 | random pass@1 | advanced pass@8 | advanced pass@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aime | 30 | 0.167 | 0.071 | 0.333 | 0.071 | 0.267 | 0.087 |
| math | 500 | 0.840 | 0.653 | 0.884 | 0.647 | 0.886 | 0.652 |
| cn_math_2024* | 30 | 0.367 | 0.104 | 0.333 | 0.079 | 0.300 | 0.071 |
| kaoyan* | 199 | 0.548 | 0.219 | 0.653 | 0.292 | 0.688 | 0.290 |
| amc* | 40 | 0.650 | 0.391 | 0.575 | 0.344 | 0.700 | 0.350 |
| minerva* | 272 | 0.452 | 0.287 | 0.471 | 0.255 | 0.463 | 0.243 |
| olympiadbench* | 675 | 0.498 | 0.277 | 0.521 | 0.274 | 0.521 | 0.288 |
| gpqa* | 198 | 0.717 | 0.294 | 0.737 | 0.230 | 0.722 | 0.224 |

`*` = counted in the leaderboard-style macro average (AIME25 not available locally; AIME24 and MATH excluded by the course definition).

## Macro average over the 6 local leaderboard benchmarks

| run | macro pass@8 | macro pass@1 | benchmarks present |
|---|---:|---:|---:|
| base | 0.5386 | 0.2619 | 6/6 |
| random | 0.5485 | 0.2456 | 6/6 |
| advanced | 0.5659 | 0.2443 | 6/6 |

## Generation length and cost (from summary.tsv)

| run | benchmark | mean gen tokens | p95 | frac at length cap | elapsed s |
|---|---|---:|---:|---:|---:|
| base | aime | 2724 | 14085 | 0.050 | 763 |
| base | math | 777 | 1245 | 0.005 | 1760 |
| base | cn_math_2024 | 1851 | 1929 | 0.025 | 622 |
| base | kaoyan | 878 | 1410 | 0.001 | 20 |
| base | amc | 1297 | 1584 | 0.016 | 555 |
| base | minerva | 734 | 1291 | 0.002 | 1275 |
| base | olympiadbench | 1476 | 1872 | 0.018 | 6302 |
| base | gpqa | 870 | 1412 | 0.004 | 986 |
| random | aime | 21148 | 32680 | 0.358 | 4621 |
| random | math | 7816 | 32657 | 0.087 | 22825 |
| random | cn_math_2024 | 17756 | 32648 | 0.200 | 3361 |
| random | kaoyan | 11018 | 32630 | 0.087 | -1 |
| random | amc | 14621 | 32689 | 0.222 | 3921 |
| random | minerva | 10573 | 32655 | 0.148 | 17961 |
| random | gpqa | 14507 | 32609 | 0.265 | 19606 |
| advanced | aime | 21381 | 32679 | 0.367 | 4661 |
| advanced | math | 7578 | 32653 | 0.084 | 21727 |
| advanced | cn_math_2024 | 18127 | 32654 | 0.217 | 3481 |
| advanced | kaoyan | 10705 | 32621 | 0.077 | 11558 |
| advanced | amc | 13471 | 32702 | 0.184 | 3381 |
| advanced | minerva | 10919 | 32668 | 0.162 | 18961 |
| advanced | gpqa | 13957 | 32592 | 0.234 | 18225 |

## Paired bootstrap over question ids (B=2000): difference = second minus first

| pair | benchmark | d pass@8 | 95% CI | d pass@1 | 95% CI |
|---|---|---:|---|---:|---|
| base -> random | aime | +0.167 | [+0.033, +0.333] | +0.000 | [-0.050, +0.046] |
| base -> random | math | +0.044 | [+0.014, +0.072] | -0.006 | [-0.026, +0.013] |
| base -> random | cn_math_2024 | -0.033 | [-0.267, +0.233] | -0.025 | [-0.104, +0.046] |
| base -> random | kaoyan | +0.106 | [+0.045, +0.166] | +0.073 | [+0.038, +0.110] |
| base -> random | amc | -0.075 | [-0.225, +0.075] | -0.047 | [-0.119, +0.022] |
| base -> random | minerva | +0.018 | [-0.026, +0.062] | -0.033 | [-0.058, -0.008] |
| base -> random | olympiadbench | +0.024 | [-0.012, +0.058] | -0.003 | [-0.019, +0.014] |
| base -> random | gpqa | +0.020 | [-0.056, +0.096] | -0.064 | [-0.102, -0.026] |
| base -> random | **macro (6 benchmarks)** | **+0.0099** | [-0.0421, +0.0618] | **-0.0163** | [-0.0370, +0.0034] |
| base -> advanced | aime | +0.100 | [-0.033, +0.233] | +0.017 | [-0.029, +0.071] |
| base -> advanced | math | +0.046 | [+0.018, +0.074] | -0.001 | [-0.022, +0.020] |
| base -> advanced | cn_math_2024 | -0.067 | [-0.267, +0.133] | -0.033 | [-0.092, +0.029] |
| base -> advanced | kaoyan | +0.141 | [+0.075, +0.201] | +0.071 | [+0.035, +0.109] |
| base -> advanced | amc | +0.050 | [-0.075, +0.175] | -0.041 | [-0.119, +0.031] |
| base -> advanced | minerva | +0.011 | [-0.033, +0.059] | -0.045 | [-0.071, -0.017] |
| base -> advanced | olympiadbench | +0.024 | [-0.009, +0.058] | +0.011 | [-0.005, +0.028] |
| base -> advanced | gpqa | +0.005 | [-0.071, +0.076] | -0.069 | [-0.104, -0.033] |
| base -> advanced | **macro (6 benchmarks)** | **+0.0273** | [-0.0191, +0.0717] | **-0.0176** | [-0.0370, +0.0019] |
| random -> advanced | aime | -0.067 | [-0.233, +0.100] | +0.017 | [-0.017, +0.054] |
| random -> advanced | math | +0.002 | [-0.018, +0.022] | +0.005 | [-0.010, +0.020] |
| random -> advanced | cn_math_2024 | -0.033 | [-0.200, +0.133] | -0.008 | [-0.046, +0.029] |
| random -> advanced | kaoyan | +0.035 | [-0.015, +0.090] | -0.002 | [-0.030, +0.026] |
| random -> advanced | amc | +0.125 | [+0.000, +0.250] | +0.006 | [-0.031, +0.047] |
| random -> advanced | minerva | -0.007 | [-0.051, +0.037] | -0.012 | [-0.032, +0.006] |
| random -> advanced | olympiadbench | +0.000 | [-0.030, +0.030] | +0.014 | [+0.001, +0.027] |
| random -> advanced | gpqa | -0.015 | [-0.081, +0.051] | -0.006 | [-0.031, +0.021] |
| random -> advanced | **macro (6 benchmarks)** | **+0.0174** | [-0.0221, +0.0574] | **-0.0013** | [-0.0130, +0.0107] |

The bootstrap captures question-sampling uncertainty for these specific trained models only; it does not capture training-seed or data-draw variance (one training run per arm).
