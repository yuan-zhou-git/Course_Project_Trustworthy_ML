# Stopping-failure inspection

| run | benchmark | samples | capped | capped & no `\boxed{}` | capped & `</think>` never reached | capped & repetitive tail | acc uncapped | acc capped |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | aime | 240 | 12 (5.0%) | 100% | 100% | 42% | 0.075 | 0.000 |
| base | math | 4000 | 22 (0.6%) | 100% | 100% | 64% | 0.657 | 0.000 |
| base | cn_math_2024 | 240 | 6 (2.5%) | 100% | 100% | 33% | 0.107 | 0.000 |
| base | kaoyan | 1592 | 2 (0.1%) | 100% | 100% | 0% | 0.219 | 0.000 |
| base | amc | 320 | 5 (1.6%) | 100% | 100% | 20% | 0.397 | 0.000 |
| base | minerva | 2176 | 5 (0.2%) | 100% | 100% | 60% | 0.288 | 0.000 |
| base | olympiadbench | 5400 | 96 (1.8%) | 99% | 100% | 36% | 0.282 | 0.000 |
| base | gpqa | 1584 | 7 (0.4%) | 100% | 100% | 86% | 0.295 | 0.000 |
| random | aime | 240 | 86 (35.8%) | 95% | 98% | 27% | 0.104 | 0.012 |
| random | math | 4000 | 349 (8.7%) | 93% | 96% | 29% | 0.707 | 0.023 |
| random | cn_math_2024 | 240 | 48 (20.0%) | 90% | 98% | 27% | 0.094 | 0.021 |
| random | kaoyan | 1592 | 139 (8.7%) | 82% | 94% | 31% | 0.317 | 0.036 |
| random | amc | 320 | 71 (22.2%) | 92% | 96% | 28% | 0.442 | 0.000 |
| random | minerva | 2176 | 321 (14.8%) | 89% | 98% | 27% | 0.296 | 0.016 |
| random | olympiadbench | 5400 | 1210 (22.4%) | 83% | 86% | 22% | 0.352 | 0.024 |
| random | gpqa | 1584 | 419 (26.5%) | 97% | 100% | 11% | 0.312 | 0.000 |
| advanced | aime | 240 | 88 (36.7%) | 95% | 97% | 26% | 0.132 | 0.011 |
| advanced | math | 4000 | 337 (8.4%) | 91% | 91% | 30% | 0.709 | 0.027 |
| advanced | cn_math_2024 | 240 | 52 (21.7%) | 94% | 100% | 38% | 0.090 | 0.000 |
| advanced | kaoyan | 1592 | 122 (7.7%) | 85% | 98% | 37% | 0.312 | 0.033 |
| advanced | amc | 320 | 59 (18.4%) | 93% | 98% | 37% | 0.429 | 0.000 |
| advanced | minerva | 2176 | 352 (16.2%) | 90% | 97% | 32% | 0.288 | 0.006 |
| advanced | olympiadbench | 5400 | 1129 (20.9%) | 92% | 96% | 25% | 0.362 | 0.009 |
| advanced | gpqa | 1584 | 370 (23.4%) | 97% | 100% | 15% | 0.291 | 0.005 |

Percentages in the middle columns are fractions of the capped samples. `</think>` never reached = the model was still in its thinking phase at the cap; repetitive = a 120-char substring repeats >= 3 times in the last 3000 chars. A capped sample without `\boxed{}` is graded wrong by construction.
