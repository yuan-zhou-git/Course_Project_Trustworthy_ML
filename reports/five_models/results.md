# Results

Runs: {'base': 'base_v1', 'random': 'random15k_v1', 'advanced': 'advanced15k_v1', 'ep5': 'advanced15k_ep5_v1', 'lp': 'lp15k_v1'}

## Per-benchmark scores (pass@8 / pass@1)

| benchmark | n | base pass@8 | base pass@1 | random pass@8 | random pass@1 | advanced pass@8 | advanced pass@1 | ep5 pass@8 | ep5 pass@1 | lp pass@8 | lp pass@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aime | 30 | 0.167 | 0.071 | 0.333 | 0.071 | 0.267 | 0.087 | 0.233 | 0.075 | 0.200 | 0.058 |
| math | 500 | 0.840 | 0.653 | 0.884 | 0.647 | 0.886 | 0.652 | 0.878 | 0.659 | 0.874 | 0.657 |
| cn_math_2024* | 30 | 0.367 | 0.104 | 0.333 | 0.079 | 0.300 | 0.071 | 0.333 | 0.104 | 0.400 | 0.108 |
| kaoyan* | 199 | 0.548 | 0.219 | 0.653 | 0.292 | 0.688 | 0.290 | 0.678 | 0.298 | 0.673 | 0.295 |
| amc* | 40 | 0.650 | 0.391 | 0.575 | 0.344 | 0.700 | 0.350 | 0.775 | 0.381 | 0.650 | 0.391 |
| minerva* | 272 | 0.452 | 0.287 | 0.471 | 0.255 | 0.463 | 0.243 | 0.456 | 0.273 | 0.463 | 0.260 |
| olympiadbench* | 675 | 0.498 | 0.277 | 0.526 | 0.278 | 0.520 | 0.288 | 0.538 | 0.301 | 0.526 | 0.290 |
| gpqa* | 198 | 0.717 | 0.294 | 0.737 | 0.230 | 0.722 | 0.224 | 0.758 | 0.258 | 0.753 | 0.245 |

`*` = counted in the leaderboard-style macro average (AIME25 not available locally; AIME24 and MATH excluded by the course definition).

## Macro average over the 6 local leaderboard benchmarks

| run | macro pass@8 | macro pass@1 | benchmarks present |
|---|---:|---:|---:|
| base | 0.5386 | 0.2619 | 6/6 |
| random | 0.5492 | 0.2463 | 6/6 |
| advanced | 0.5656 | 0.2443 | 6/6 |
| ep5 | 0.5897 | 0.2691 | 6/6 |
| lp | 0.5775 | 0.2648 | 6/6 |

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
| ep5 | aime | 21283 | 32686 | 0.354 | 5901 |
| ep5 | cn_math_2024 | 17108 | 32670 | 0.167 | 4081 |
| ep5 | amc | 12819 | 32681 | 0.138 | 4101 |
| lp | aime | 19498 | 32678 | 0.258 | 3921 |
| lp | cn_math_2024 | 15482 | 32635 | 0.125 | 2801 |
| lp | amc | 11819 | 32665 | 0.138 | 2901 |

## Paired bootstrap over question ids (B=2000): difference = second minus first

| pair | benchmark | d pass@8 | 95% CI | d pass@1 | 95% CI |
|---|---|---:|---|---:|---|
| base -> random | aime | +0.167 | [+0.033, +0.333] | +0.000 | [-0.050, +0.046] |
| base -> random | math | +0.044 | [+0.014, +0.072] | -0.006 | [-0.026, +0.013] |
| base -> random | cn_math_2024 | -0.033 | [-0.267, +0.233] | -0.025 | [-0.104, +0.046] |
| base -> random | kaoyan | +0.106 | [+0.045, +0.166] | +0.073 | [+0.038, +0.110] |
| base -> random | amc | -0.075 | [-0.225, +0.075] | -0.047 | [-0.119, +0.022] |
| base -> random | minerva | +0.018 | [-0.026, +0.062] | -0.033 | [-0.058, -0.008] |
| base -> random | olympiadbench | +0.028 | [-0.006, +0.061] | +0.001 | [-0.015, +0.018] |
| base -> random | gpqa | +0.020 | [-0.056, +0.096] | -0.064 | [-0.102, -0.026] |
| base -> random | **macro (6 benchmarks)** | **+0.0107** | [-0.0424, +0.0624] | **-0.0157** | [-0.0361, +0.0039] |
| base -> advanced | aime | +0.100 | [-0.033, +0.233] | +0.017 | [-0.029, +0.071] |
| base -> advanced | math | +0.046 | [+0.018, +0.074] | -0.001 | [-0.022, +0.020] |
| base -> advanced | cn_math_2024 | -0.067 | [-0.267, +0.133] | -0.033 | [-0.092, +0.029] |
| base -> advanced | kaoyan | +0.141 | [+0.075, +0.201] | +0.071 | [+0.035, +0.109] |
| base -> advanced | amc | +0.050 | [-0.075, +0.175] | -0.041 | [-0.119, +0.031] |
| base -> advanced | minerva | +0.011 | [-0.033, +0.059] | -0.045 | [-0.071, -0.017] |
| base -> advanced | olympiadbench | +0.022 | [-0.009, +0.055] | +0.011 | [-0.006, +0.028] |
| base -> advanced | gpqa | +0.005 | [-0.071, +0.076] | -0.069 | [-0.104, -0.033] |
| base -> advanced | **macro (6 benchmarks)** | **+0.0271** | [-0.0196, +0.0714] | **-0.0176** | [-0.0371, +0.0018] |
| base -> ep5 | aime | +0.067 | [-0.067, +0.200] | +0.004 | [-0.029, +0.037] |
| base -> ep5 | math | +0.038 | [+0.008, +0.068] | +0.006 | [-0.013, +0.025] |
| base -> ep5 | cn_math_2024 | -0.033 | [-0.233, +0.167] | +0.000 | [-0.075, +0.071] |
| base -> ep5 | kaoyan | +0.131 | [+0.070, +0.191] | +0.079 | [+0.041, +0.114] |
| base -> ep5 | amc | +0.125 | [+0.000, +0.250] | -0.009 | [-0.075, +0.062] |
| base -> ep5 | minerva | +0.004 | [-0.044, +0.051] | -0.015 | [-0.041, +0.011] |
| base -> ep5 | olympiadbench | +0.040 | [+0.006, +0.073] | +0.025 | [+0.009, +0.043] |
| base -> ep5 | gpqa | +0.040 | [-0.030, +0.111] | -0.036 | [-0.072, +0.000] |
| base -> ep5 | **macro (6 benchmarks)** | **+0.0511** | [+0.0087, +0.0950] | **+0.0072** | [-0.0113, +0.0266] |
| base -> lp | aime | +0.033 | [-0.100, +0.133] | -0.012 | [-0.063, +0.037] |
| base -> lp | math | +0.034 | [+0.002, +0.066] | +0.004 | [-0.014, +0.023] |
| base -> lp | cn_math_2024 | +0.033 | [-0.133, +0.200] | +0.004 | [-0.067, +0.071] |
| base -> lp | kaoyan | +0.126 | [+0.055, +0.191] | +0.075 | [+0.041, +0.111] |
| base -> lp | amc | +0.000 | [-0.150, +0.150] | +0.000 | [-0.078, +0.075] |
| base -> lp | minerva | +0.011 | [-0.033, +0.062] | -0.027 | [-0.053, -0.003] |
| base -> lp | olympiadbench | +0.028 | [-0.006, +0.061] | +0.013 | [-0.004, +0.031] |
| base -> lp | gpqa | +0.035 | [-0.035, +0.116] | -0.049 | [-0.083, -0.015] |
| base -> lp | **macro (6 benchmarks)** | **+0.0389** | [-0.0043, +0.0816] | **+0.0029** | [-0.0169, +0.0234] |
| random -> advanced | aime | -0.067 | [-0.233, +0.100] | +0.017 | [-0.017, +0.054] |
| random -> advanced | math | +0.002 | [-0.018, +0.024] | +0.005 | [-0.010, +0.021] |
| random -> advanced | cn_math_2024 | -0.033 | [-0.200, +0.133] | -0.008 | [-0.042, +0.029] |
| random -> advanced | kaoyan | +0.035 | [-0.020, +0.085] | -0.002 | [-0.030, +0.025] |
| random -> advanced | amc | +0.125 | [+0.000, +0.250] | +0.006 | [-0.034, +0.047] |
| random -> advanced | minerva | -0.007 | [-0.051, +0.037] | -0.012 | [-0.032, +0.008] |
| random -> advanced | olympiadbench | -0.006 | [-0.036, +0.022] | +0.010 | [-0.003, +0.021] |
| random -> advanced | gpqa | -0.015 | [-0.081, +0.056] | -0.006 | [-0.033, +0.021] |
| random -> advanced | **macro (6 benchmarks)** | **+0.0164** | [-0.0237, +0.0558] | **-0.0020** | [-0.0137, +0.0103] |
| random -> ep5 | aime | -0.100 | [-0.200, +0.000] | +0.004 | [-0.021, +0.033] |
| random -> ep5 | math | -0.006 | [-0.028, +0.018] | +0.012 | [-0.003, +0.028] |
| random -> ep5 | cn_math_2024 | +0.000 | [-0.167, +0.200] | +0.025 | [-0.021, +0.075] |
| random -> ep5 | kaoyan | +0.025 | [-0.020, +0.070] | +0.006 | [-0.021, +0.033] |
| random -> ep5 | amc | +0.200 | [+0.050, +0.375] | +0.037 | [-0.012, +0.088] |
| random -> ep5 | minerva | -0.015 | [-0.055, +0.022] | +0.018 | [-0.001, +0.035] |
| random -> ep5 | olympiadbench | +0.012 | [-0.016, +0.039] | +0.023 | [+0.012, +0.035] |
| random -> ep5 | gpqa | +0.020 | [-0.045, +0.086] | +0.028 | [-0.001, +0.057] |
| random -> ep5 | **macro (6 benchmarks)** | **+0.0404** | [-0.0037, +0.0861] | **+0.0229** | [+0.0092, +0.0375] |
| random -> lp | aime | -0.133 | [-0.267, -0.033] | -0.012 | [-0.046, +0.025] |
| random -> lp | math | -0.010 | [-0.032, +0.012] | +0.010 | [-0.005, +0.025] |
| random -> lp | cn_math_2024 | +0.067 | [-0.133, +0.233] | +0.029 | [-0.017, +0.079] |
| random -> lp | kaoyan | +0.020 | [-0.025, +0.065] | +0.003 | [-0.021, +0.026] |
| random -> lp | amc | +0.075 | [-0.050, +0.200] | +0.047 | [-0.003, +0.100] |
| random -> lp | minerva | -0.007 | [-0.051, +0.033] | +0.006 | [-0.011, +0.022] |
| random -> lp | olympiadbench | +0.000 | [-0.030, +0.030] | +0.012 | [-0.001, +0.024] |
| random -> lp | gpqa | +0.015 | [-0.051, +0.081] | +0.015 | [-0.011, +0.042] |
| random -> lp | **macro (6 benchmarks)** | **+0.0283** | [-0.0114, +0.0693] | **+0.0185** | [+0.0054, +0.0323] |
| advanced -> ep5 | aime | -0.033 | [-0.133, +0.067] | -0.012 | [-0.038, +0.008] |
| advanced -> ep5 | math | -0.008 | [-0.028, +0.012] | +0.008 | [-0.007, +0.022] |
| advanced -> ep5 | cn_math_2024 | +0.033 | [-0.133, +0.200] | +0.033 | [-0.017, +0.092] |
| advanced -> ep5 | kaoyan | -0.010 | [-0.060, +0.040] | +0.008 | [-0.021, +0.036] |
| advanced -> ep5 | amc | +0.075 | [-0.025, +0.200] | +0.031 | [-0.019, +0.075] |
| advanced -> ep5 | minerva | -0.007 | [-0.048, +0.033] | +0.030 | [+0.013, +0.047] |
| advanced -> ep5 | olympiadbench | +0.018 | [-0.010, +0.046] | +0.014 | [+0.002, +0.025] |
| advanced -> ep5 | gpqa | +0.035 | [-0.035, +0.106] | +0.033 | [+0.005, +0.061] |
| advanced -> ep5 | **macro (6 benchmarks)** | **+0.0240** | [-0.0133, +0.0636] | **+0.0249** | [+0.0103, +0.0394] |
| advanced -> lp | aime | -0.067 | [-0.200, +0.067] | -0.029 | [-0.058, -0.004] |
| advanced -> lp | math | -0.012 | [-0.034, +0.010] | +0.005 | [-0.009, +0.020] |
| advanced -> lp | cn_math_2024 | +0.100 | [-0.033, +0.267] | +0.038 | [-0.012, +0.092] |
| advanced -> lp | kaoyan | -0.015 | [-0.070, +0.040] | +0.004 | [-0.020, +0.030] |
| advanced -> lp | amc | -0.050 | [-0.200, +0.075] | +0.041 | [+0.003, +0.078] |
| advanced -> lp | minerva | +0.000 | [-0.051, +0.048] | +0.017 | [-0.002, +0.036] |
| advanced -> lp | olympiadbench | +0.006 | [-0.019, +0.033] | +0.002 | [-0.010, +0.014] |
| advanced -> lp | gpqa | +0.030 | [-0.040, +0.096] | +0.021 | [-0.008, +0.049] |
| advanced -> lp | **macro (6 benchmarks)** | **+0.0119** | [-0.0251, +0.0504] | **+0.0205** | [+0.0075, +0.0336] |
| ep5 -> lp | aime | -0.033 | [-0.133, +0.067] | -0.017 | [-0.050, +0.008] |
| ep5 -> lp | math | -0.004 | [-0.030, +0.022] | -0.002 | [-0.018, +0.013] |
| ep5 -> lp | cn_math_2024 | +0.067 | [-0.067, +0.200] | +0.004 | [-0.033, +0.042] |
| ep5 -> lp | kaoyan | -0.005 | [-0.055, +0.045] | -0.003 | [-0.030, +0.022] |
| ep5 -> lp | amc | -0.125 | [-0.275, +0.025] | +0.009 | [-0.047, +0.072] |
| ep5 -> lp | minerva | +0.007 | [-0.033, +0.044] | -0.012 | [-0.030, +0.005] |
| ep5 -> lp | olympiadbench | -0.012 | [-0.040, +0.018] | -0.011 | [-0.024, +0.000] |
| ep5 -> lp | gpqa | -0.005 | [-0.076, +0.066] | -0.013 | [-0.041, +0.015] |
| ep5 -> lp | **macro (6 benchmarks)** | **-0.0122** | [-0.0489, +0.0250] | **-0.0044** | [-0.0178, +0.0088] |

The bootstrap captures question-sampling uncertainty for these specific trained models only; it does not capture training-seed or data-draw variance (one training run per arm).


## Shard sources used per (run, benchmark)

Rule: each question index is taken from the narrowest result shard covering it (ties: earliest start). Counts = questions taken from each file.

- advanced15k_ep5_v1/gpqa: `test_qwen-instruct_t0.6_k8_s0_e99.jsonl`: 99, `test_qwen-instruct_t0.6_k8_s99_e198.jsonl`: 99
- advanced15k_ep5_v1/kaoyan: `test_qwen-instruct_t0.6_k8_s0_e100.jsonl`: 100, `test_qwen-instruct_t0.6_k8_s100_e199.jsonl`: 99
- advanced15k_ep5_v1/math: `test_qwen-instruct_t0.6_k8_s0_e125.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s125_e250.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s250_e375.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s375_e500.jsonl`: 125
- advanced15k_ep5_v1/minerva: `test_qwen-instruct_t0.6_k8_s0_e136.jsonl`: 136, `test_qwen-instruct_t0.6_k8_s136_e272.jsonl`: 136
- advanced15k_ep5_v1/olympiadbench: `test_qwen-instruct_t0.6_k8_s0_e85.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s169_e254.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s254_e338.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s338_e423.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s423_e507.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s507_e591.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s591_e675.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s85_e169.jsonl`: 84
- advanced15k_v1/olympiadbench: `test_qwen-instruct_t0.6_k8_s0_e85.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s169_e254.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s254_e338.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s338_e423.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s423_e507.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s507_e591.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s591_e675.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s85_e169.jsonl`: 84
- lp15k_v1/gpqa: `test_qwen-instruct_t0.6_k8_s0_e99.jsonl`: 99, `test_qwen-instruct_t0.6_k8_s99_e198.jsonl`: 99
- lp15k_v1/kaoyan: `test_qwen-instruct_t0.6_k8_s0_e100.jsonl`: 100, `test_qwen-instruct_t0.6_k8_s100_e199.jsonl`: 99
- lp15k_v1/math: `test_qwen-instruct_t0.6_k8_s0_e125.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s125_e250.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s250_e375.jsonl`: 125, `test_qwen-instruct_t0.6_k8_s375_e500.jsonl`: 125
- lp15k_v1/minerva: `test_qwen-instruct_t0.6_k8_s0_e136.jsonl`: 136, `test_qwen-instruct_t0.6_k8_s136_e272.jsonl`: 136
- lp15k_v1/olympiadbench: `test_qwen-instruct_t0.6_k8_s0_e85.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s169_e254.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s254_e338.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s338_e423.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s423_e507.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s507_e591.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s591_e675.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s85_e169.jsonl`: 84
- random15k_v1/olympiadbench: `test_qwen-instruct_t0.6_k8_s0_e169.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s0_e85.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s169_e254.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s254_e338.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s338_e423.jsonl`: 85, `test_qwen-instruct_t0.6_k8_s423_e507.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s507_e591.jsonl`: 84, `test_qwen-instruct_t0.6_k8_s591_e675.jsonl`: 84