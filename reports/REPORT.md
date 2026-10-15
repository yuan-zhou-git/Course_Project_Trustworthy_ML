# Reasoning SFT of Qwen2.5-3B-Instruct: random vs. selected training data

Course project report (ECE, Virginia Tech, Fall 2026). Repo: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML. Deadline Oct 20, 2026.

## 1. Summary

We fine-tuned Qwen2.5-3B-Instruct twice on 15K samples from the 100K AceReason-1.1-SFT subset, once with a uniform random
subset and once with a cheap, label-free selected subset (math-only, deduplicated, length-stratified), using identical
hyperparameters, and evaluated the base model and both fine-tuned models on the 8 course benchmarks with the official
protocol (8 samples, temperature 0.6, top-p 0.95, 32,768 max tokens).

**Headline (fill from `reports/results.md`):** TODO macro pass@8 base / random / selected = ... ; paired bootstrap CI for
selected minus random = ... .

## 2. Setup

| item | value |
|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct`, snapshot `aa8e7253` |
| training pool | `redsgnaoh/acereason11_100k` (100,000 rows; 67,432 math / 32,568 code; sha256 in `pool_stats.json`) |
| frameworks | LLaMA-Factory 0.9.4 (course copy), DeepSpeed 0.16.8 (ZeRO-3, no offload), FlashAttention 2.8.3, Liger kernel 0.8.4, torch 2.8.0+cu128; vLLM 0.31.0 for evaluation |
| hardware | Virginia Tech ARC Falcon: 4x NVIDIA L40S (48 GB) per training run; 1x L40S or A30 per evaluation job |
| hyperparameters (both arms) | full-parameter SFT, bf16, lr 5e-6, cosine, warmup 0, 3 epochs, global batch 8 (4 GPUs x 1 x grad-accum 2), cutoff 16,384 tokens, response-only loss, `template: qwen` |
| training cost | random: 5,625 steps, 338.8M tokens, 9 h 09 min; selected: 5,625 steps, 322.9M tokens, 8 h 35 min (~36 GPU-hours each) |
| final train loss | random 0.581; selected 0.530 |

Tokens are counted on the full chat-template encoding (system + user + assistant + end-of-turn). The Liger fused
linear cross-entropy was required to fit 16K-token samples on 44 GB GPUs; without it the first long sample OOMed.
The label mask was verified on a decoded example before the long runs (prompt tokens masked, labels start at `<think>`).

## 3. Data selection

### 3.1 Random 15K (required baseline)
`random.Random(20261015).sample` over the whole pool, no filtering. 113.4M training tokens per epoch; 556 rows are longer
than the cutoff and get truncated (their final answer is lost in training).

### 3.2 Selected 15K
Motivation: the benchmarks are all math and science, so code traces (33% of the pool) are off-target; AceReason contains
many repeated prompts (15,906 exact duplicates); and 2.3% of rows do not fit the cutoff. Filters, in order:

1. `extra_info.category == "math"` (removes 32,568)
2. exact-duplicate prompts removed, first occurrence kept (removes 7,730)
3. full conversation <= 16,384 tokens (removes 2,292); empty outputs or unbalanced `<think>` tags (0 found)
4. 57,410 eligible rows; stratified sampling by source x total-token band (<=4K, 4-8K, 8-16K) with proportional quotas,
   seed 20261015: NuminaMath-CoT 1162 / 988 / 957, OpenMathReasoning 3091 / 4174 / 4628.

Result: 107.6M training tokens per epoch (5% less compute than random), median sample 6.5K tokens, nothing truncated.
2,252 rows are shared with the random set. The method needs no model inference, embeddings or answer keys and is
reproducible from `scripts/prep_data.py` and the frozen pool file. Design references: OpenThoughts (question difficulty
and length filtering, source mixing), AceReason-Nemotron 1.1 (prompt diversity), LIMO (quality over quantity); we
deliberately did not use model-based difficulty scoring or LIMOPro pruning in this first pass (cost and added failure
surface under a 5-day budget).

Benchmark overlap: 6 pool prompts match a benchmark question verbatim (3 MATH, 2 OlympiadBench, 1 AMC). One landed in
each 15K set and was kept in both arms rather than sanitizing one arm only. Effect on any score is at most one question.

## 4. Evaluation protocol

Course `eval/` harness: vLLM generation with the system prompt "Please reason step by step, and put your final answer
within \boxed{}", last `\boxed{}` extracted, SymPy-based equivalence grading. pass@1 = mean per-sample accuracy;
pass@8 = any of 8 samples correct. Leaderboard-style macro average over the 6 locally available leaderboard benchmarks
(cn_math_2024, kaoyan, amc, minerva, olympiadbench, gpqa); AIME25 is not available locally; AIME24 and MATH are reported
but excluded, following the course definition.

Repairs we made around the harness (no change to the protocol): strict failure handling, immutable run directories with a
manifest, an 8-responses-per-question completeness check, a watchdog that kills the evaluation process if it hangs after
printing its scores (observed twice; the grader's thread pool blocks interpreter exit), the seed forwarded to vLLM, and
pass@k recomputed from the per-sample correctness file. MATH-500 contains 500 records (the README's 499 is a line count).

## 5. Results

TODO: paste the per-benchmark table, macro averages, generation lengths and the paired bootstrap from `reports/results.md`.

Base model (1x A30): aime 0.167/0.071, amc 0.650/0.391, cn_math_2024 0.367/0.104, gpqa 0.717/0.294, kaoyan 0.548/0.219,
math 0.840/0.653, minerva 0.452/0.287, olympiadbench 0.498/0.277 (pass@8/pass@1); macro-6 pass@8 0.539. These match the
Fall-2025 baseline row within benchmark noise.

## 6. Analysis

TODO after results: per-benchmark deltas, GPQA regression check, generation length change (base ~0.7-2.7K tokens per
answer; fine-tuned models expected much longer), fraction of answers hitting the 32K cap, cost per evaluation.
What the single-run design can and cannot claim (one training run per arm; the bootstrap covers question sampling only).

## 7. Reproducibility and artifacts

- Models: `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k`, `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k`
- Datasets: `yuan-zhou-hf/acereason-random15k`, `yuan-zhou-hf/acereason-advanced15k` (json + manifest + pool stats)
- Code: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML (`scripts/prep_data.py`, `slurm/*.slurm`, `yamls/*.yaml`, `scripts/eval_summary.py`,
  `scripts/aggregate_results.py`), environment freezes `envs/*_freeze.txt`, W&B project `reasoning-sft`.
- Slurm job ids (Falcon): training 645899 (random), 645900 (selected); base eval 645832/646490; fine-tuned evals 646668-646673.
