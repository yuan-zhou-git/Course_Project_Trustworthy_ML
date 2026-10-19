# Reasoning SFT of Qwen2.5-3B-Instruct: random vs. selected training data

Course project report (ECE, Virginia Tech, Fall 2026). Repo: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML.
Assignment: [Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM). Evaluated 2026-10-15/16.

## 1. Summary

We fine-tuned Qwen2.5-3B-Instruct twice on 15K samples from the 100K AceReason-1.1-SFT subset, once with a uniform random
subset (the required baseline) and once with a cheap, label-free selected subset (math-only, deduplicated, length-stratified),
with identical hyperparameters, and evaluated the base model and both fine-tuned models on the 8 course benchmarks under the
official protocol (8 samples, temperature 0.6, top-p 0.95, 32,768 max tokens).

| model | macro pass@8 (6 leaderboard benchmarks) | macro pass@1 |
|---|---:|---:|
| Qwen2.5-3B-Instruct (base) | 0.539 | 0.262 |
| SFT, random 15K | 0.549 | 0.246 |
| SFT, selected 15K | **0.566** | 0.244 |

Paired question-level bootstrap (B = 2000), macro pass@8: selected minus random **+0.016** [-0.024, +0.056]; selected minus base
+0.027 [-0.020, +0.071]; random minus base +0.011 [-0.042, +0.062]. In words: the selected set had the higher observed pass@8 in
this controlled run, with the largest and statistically clear gains over the base model on Kaoyan and MATH-500; the difference
between the two data-selection arms is concentrated in AMC (five questions) and is within question-sampling noise. Both SFT
models lose pass@1 on Minerva and GPQA, coinciding with 15 to 27% of their samples hitting the 32K token cap. With one training
run per arm we claim "higher observed pass@8 in this run", not "reliably superior".

**Follow-up arms (Oct 17-18, after the comparison above was frozen; sections 6c and 6d).**

| model | macro pass@8 | macro pass@1 |
|---|---:|---:|
| SFT, learning-progress 15K (3 epochs; selection rule fixed before training) | 0.578 | **0.265** |
| SFT, selected 15K trained for 5 epochs instead of 3 | **0.590** | **0.269** |

A student-dependent selector (learning progress: Qwen's loss on a solution before minus after a short probe fine-tune) is the
best equal-budget arm, with fewer training tokens than random: +0.028 [-0.011, +0.069] pass@8 and **+0.019 [+0.005, +0.032]**
pass@1 over random, +0.021 [+0.008, +0.034] pass@1 over the first selected set. Training the first selected set for 5 epochs
gives the best model overall and the only pass@8 gain over the base model whose interval excludes zero (+0.051 [+0.009,
+0.095]); that is a training-budget result, not an equal-budget data comparison (random was not trained to 5 epochs). Both
gains run through the mechanism identified in the main comparison: fewer samples cut off at the 32K cap, while the accuracy of
finished samples barely moves.

## 2. Setup

| item | value |
|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct`, snapshot `aa8e7253` |
| training pool | `redsgnaoh/acereason11_100k` (100,000 rows; 67,432 math / 32,568 code; sha256 in `data/pool_stats.json`) |
| frameworks | LLaMA-Factory 0.9.4 (course copy), DeepSpeed 0.16.8 (ZeRO-3, no offload), FlashAttention 2.8.3, Liger kernel 0.8.4, torch 2.8.0+cu128; vLLM 0.31.0 for evaluation |
| hardware | Virginia Tech ARC Falcon: 4x NVIDIA L40S (48 GB) per training run; 1x L40S or A30 per evaluation job |
| hyperparameters (both arms) | full-parameter SFT, bf16, lr 5e-6, cosine, warmup 0, 3 epochs, global batch 8 (4 GPUs x micro-batch 1 x grad-accum 2), cutoff 16,384 tokens, response-only loss, `template: qwen` |
| training cost | random: 5,625 steps, 338.8M tokens, 9 h 09 min; selected: 5,625 steps, 322.9M tokens, 8 h 35 min (~36 GPU-hours each) |
| final train loss | random 0.581; selected 0.530 |
| evaluation cost | base ~3.5 GPU-h (A30); each fine-tuned model ~45-50 GPU-h (L40S/A30 mix), because answers are 10-20x longer |

Tokens are counted on the full chat-template encoding (system + user + assistant + end-of-turn). The Liger fused linear
cross-entropy was required to fit 16K-token samples on 44 GB GPUs; without it the first long sample ran out of memory. The label
mask was verified on a decoded example before the long runs (prompt tokens masked, labels start at `<think>`). W&B project
`reasoning-sft` holds both loss curves (`training_loss.png` is also in `models/<arm>/`).

## 3. Data selection

### 3.1 Random 15K (required baseline)
`random.Random(20261015).sample` over the whole pool, no filtering. 113.4M training tokens per epoch; 556 rows exceed the cutoff
and are truncated (their final answer is lost in training).

### 3.2 Selected 15K
Motivation: the benchmarks are all math and science, so code traces (33% of the pool) are off-target; AceReason contains many
repeated prompts (15,906 exact duplicates); and 3.5% of pool rows (3,452) do not fit the cutoff. Filters, in order:

1. `extra_info.category == "math"` (removes 32,568)
2. exact-duplicate prompts removed, first occurrence kept (removes 7,730)
3. full conversation <= 16,384 tokens (removes 2,292); empty outputs or unbalanced `<think>` tags (0 found)
4. 57,410 eligible rows; stratified sampling by source x total-token band (<=4K, 4-8K, 8-16K) with proportional quotas,
   seed 20261015: NuminaMath-CoT 1162 / 988 / 957, OpenMathReasoning 3091 / 4174 / 4628.

Result: 107.6M training tokens per epoch (5% less than random), median sample 6.5K tokens, nothing truncated; 2,252 rows are
shared with the random set. The method needs no model inference, embeddings or answer keys and is reproducible from
`scripts/prep_data.py` and the frozen pool file. Design references: OpenThoughts (question filtering, source mixing),
AceReason-Nemotron 1.1 (prompt diversity), LIMO (quality over quantity). We deliberately did not use model-based difficulty
scoring or LIMOPro pruning in this pass (cost and added failure surface under a 5-day budget).

Benchmark overlap: 6 pool prompts match a benchmark question verbatim (3 MATH, 2 OlympiadBench, 1 AMC). One landed in each
15K set; both were kept rather than sanitizing one arm only. Effect on any score is at most one question.

## 4. Evaluation protocol

Course `eval/` harness: vLLM generation with the system prompt "Please reason step by step, and put your final answer within
\boxed{}", last `\boxed{}` extracted, SymPy-based equivalence grading. pass@1 = mean per-sample accuracy; pass@8 = any of 8
samples correct. Leaderboard-style macro average over the 6 locally available leaderboard benchmarks (cn_math_2024, kaoyan,
amc, minerva, olympiadbench, gpqa); AIME25 is not available locally; AIME24 and MATH are reported but excluded, following the
course leaderboard definition.

Repairs around the harness (protocol unchanged): strict failure handling, immutable run directories with a manifest, an
8-responses-per-question completeness check, a watchdog that kills the evaluation process if it hangs after printing its scores
(observed repeatedly; the grader's thread pool blocks interpreter exit), the seed forwarded to vLLM (one-line patch,
`patches/eval_seed.patch`), pass@k recomputed from the per-sample correctness file, and index-range sharding of OlympiadBench
(the course script writes nothing until a whole benchmark finishes, and a full run of the fine-tuned models was projected at
13-27 h on one GPU). Shards are merged by question index in `scripts/aggregate_results.py`. MATH-500 contains 500 records
(the README's 499 is a line count).

## 5. Results

### 5.1 Per-benchmark scores (pass@8 / pass@1)

| benchmark | n | base | random 15K | selected 15K |
|---|---:|---|---|---|
| aime (AIME24) | 30 | 0.167 / 0.071 | 0.333 / 0.071 | 0.267 / 0.087 |
| math (MATH-500) | 500 | 0.840 / 0.653 | 0.884 / 0.647 | 0.886 / 0.652 |
| cn_math_2024* | 30 | 0.367 / 0.104 | 0.333 / 0.079 | 0.300 / 0.071 |
| kaoyan* | 199 | 0.548 / 0.219 | 0.653 / 0.292 | 0.688 / 0.290 |
| amc* | 40 | 0.650 / 0.391 | 0.575 / 0.344 | 0.700 / 0.350 |
| minerva* | 272 | 0.452 / 0.287 | 0.471 / 0.255 | 0.463 / 0.243 |
| olympiadbench* | 675 | 0.498 / 0.277 | 0.526 / 0.278 | 0.520 / 0.288 |
| gpqa* | 198 | 0.717 / 0.294 | 0.737 / 0.230 | 0.722 / 0.224 |
| **macro (6 starred)** | | **0.539 / 0.262** | **0.549 / 0.246** | **0.566 / 0.244** |

### 5.2 Paired bootstrap over question ids (B = 2000, 95% CI), difference = second minus first

| pair | benchmark | d pass@8 | CI | d pass@1 | CI |
|---|---|---:|---|---:|---|
| base -> random | kaoyan | +0.106 | [+0.045, +0.166] | +0.073 | [+0.038, +0.110] |
| base -> random | math | +0.044 | [+0.014, +0.072] | -0.006 | [-0.026, +0.013] |
| base -> random | aime | +0.167 | [+0.033, +0.333] | +0.000 | [-0.050, +0.046] |
| base -> random | minerva | +0.018 | [-0.026, +0.062] | -0.033 | [-0.058, -0.008] |
| base -> random | gpqa | +0.020 | [-0.056, +0.096] | -0.064 | [-0.102, -0.026] |
| base -> random | olympiadbench | +0.028 | [-0.006, +0.061] | +0.001 | [-0.015, +0.018] |
| base -> random | **macro** | **+0.011** | [-0.042, +0.062] | **-0.016** | [-0.036, +0.004] |
| base -> selected | kaoyan | +0.141 | [+0.075, +0.201] | +0.071 | [+0.035, +0.109] |
| base -> selected | math | +0.046 | [+0.018, +0.074] | -0.001 | [-0.022, +0.020] |
| base -> selected | minerva | +0.011 | [-0.033, +0.059] | -0.045 | [-0.071, -0.017] |
| base -> selected | gpqa | +0.005 | [-0.071, +0.076] | -0.069 | [-0.104, -0.033] |
| base -> selected | olympiadbench | +0.022 | [-0.009, +0.055] | +0.011 | [-0.006, +0.028] |
| base -> selected | **macro** | **+0.027** | [-0.020, +0.071] | **-0.018** | [-0.037, +0.002] |
| random -> selected | amc | +0.125 | [+0.000, +0.250] | +0.006 | [-0.031, +0.047] |
| random -> selected | kaoyan | +0.035 | [-0.015, +0.090] | -0.002 | [-0.030, +0.026] |
| random -> selected | olympiadbench | -0.006 | [-0.034, +0.022] | +0.010 | [-0.002, +0.022] |
| random -> selected | aime | -0.067 | [-0.233, +0.100] | +0.017 | [-0.017, +0.054] |
| random -> selected | **macro** | **+0.016** | [-0.024, +0.056] | **-0.002** | [-0.014, +0.010] |

Full table for all 8 benchmarks and both metrics: `reports/results.md` (`results.json` holds the numbers;
`per_question_correctness.json` holds the 8 correctness flags per question so the bootstrap can be recomputed). OlympiadBench
was evaluated in index shards; each question is taken from the narrowest shard covering it (rule and file list at the end of
`results.md`), and every record's question text was verified against the benchmark file at its index.

### 5.3 Generation length and cost

| run | mean generated tokens (range over benchmarks) | fraction of samples at the 32,768 cap | eval GPU-hours |
|---|---|---|---:|
| base | 0.7K (minerva) to 2.7K (aime) | 0.1% to 5% | ~3.5 (A30) |
| random 15K | 7.8K (math) to 21.1K (aime); olympiadbench 14.5K | 8.7% (math, kaoyan) to 36% (aime); olympiadbench 22% | ~50 |
| selected 15K | 7.6K (math) to 21.4K (aime); olympiadbench 13.9K | 7.7% (kaoyan) to 37% (aime); olympiadbench 21% | ~49 |

### 5.4 What the capped samples look like (`reports/stopping_failures.md`)

| run | capped samples (all 8 benchmarks) | of those: still inside `<think>` | repetitive tail | per-sample accuracy, uncapped vs capped |
|---|---:|---:|---:|---|
| base | 155 / 15,552 (1.0%) | 100% | 20-86% | e.g. MATH 0.657 vs 0.000 |
| random 15K | 2,643 / 15,552 (17.0%) | 86-100% | 11-31% | MATH 0.707 vs 0.023; Minerva 0.296 vs 0.016; GPQA 0.312 vs 0.000 |
| selected 15K | 2,509 / 15,552 (16.1%) | 91-100% | 15-38% | MATH 0.709 vs 0.027; Minerva 0.288 vs 0.006; GPQA 0.291 vs 0.005 |

Most capped fine-tuned samples are unfinished reasoning (the `</think>` tag is never reached), not degenerate loops; a minority
(11 to 38%) end in a repeating tail. On the samples that do finish, both fine-tuned models match or beat the base model's
per-sample accuracy on every benchmark where pass@1 fell, so the pass@1 losses are a stopping problem rather than a reasoning one.
Whether a larger generation budget would convert those samples into correct answers was not tested.

## 6. Analysis

- **What SFT changed.** Both fine-tuned models write DeepSeek-R1-style `<think>` traces that are 10 to 20 times longer than
  the base model's answers. pass@8 rises on 7 of 8 benchmarks for the selected model (all but the 30-question CN_MATH set, whose
  interval spans zero), with clear gains on Kaoyan (+0.14) and MATH-500 (+0.05). pass@1 falls on Minerva and GPQA for both SFT
  models; these are exactly the benchmarks where 15 to 27% of samples hit the 32K cap, and a capped sample without a final
  `\boxed{}` is graded wrong by construction. Whether a longer generation budget would recover that loss is untested here (the
  course protocol fixes 32,768 tokens); `reports/stopping_failures.md` classifies the capped samples (still in the thinking phase,
  repetitive loop, or missing conclusion).
- **Selected vs random.** The selected set wins the leaderboard metric by +0.016 macro pass@8, but the advantage is narrow in
  origin: five additional AMC questions (+0.125 on a 40-question set) contribute about +0.021 to the macro average on their own,
  while the other five included benchmarks net about -0.004 (Kaoyan +0.035, CN_MATH -0.033, Minerva -0.007, OlympiadBench -0.006,
  GPQA -0.015). The interval [-0.024, +0.056] leaves superiority unresolved. On pass@1 the two are indistinguishable (-0.002).
  The selected set used 5% fewer training tokens. Because the recipe combines three filters and stratification, this run cannot
  attribute the difference to any single filter.
- **Against the Fall-2025 leaderboard.** That table averages 7 benchmarks including AIME25, which we do not have, so numbers
  are not directly comparable; our base reproduces the listed baseline row within benchmark noise (e.g. MATH 0.840 exactly),
  and the fine-tuned models' per-benchmark pass@8 values sit in the range of last year's top entries on Kaoyan and MATH.
- **Limits.** One training run per arm; the bootstrap covers question sampling only, not training-seed or data-draw variance.
  The historical spread between two random draws (4.1 points) is larger than our selected-minus-random difference, so a second
  seed per arm would be the next experiment. AIME/CN_MATH have 30 questions each and move by 0.033 per question.

## 6b. Follow-up experiments (after the main comparison was frozen)

Both were run after the audit of the main results and are reported separately; neither changes the main comparison.

**Second generation seed on AMC (seed 1, all other settings official).** AMC carried most of the selected-vs-random gap.

| model | seed 0 pass@8 / pass@1 | seed 1 pass@8 / pass@1 | two-seed mean pass@8 |
|---|---|---|---:|
| base | 0.650 / 0.391 | 0.700 / 0.419 | 0.675 |
| random 15K | 0.575 / 0.344 | 0.700 / 0.356 | 0.638 |
| selected 15K | 0.700 / 0.350 | 0.750 / 0.394 | 0.725 |

The selected model leads on both seeds, but the gap over random shrinks from five questions to two, and the random model alone
moved five questions between seeds. The direction of the AMC advantage is consistent; its size is within sampling noise on a
40-question set. Run dirs `eval_outputs/*_seed1`.

**Checkpoint screening on a held-out development panel (no benchmark data).** 200 math questions drawn from pool rows that are
in neither 15K set (36,491 eligible), stratified by teacher-response length, scored against the teacher's final `\boxed{}` answer
with the course grader (teacher agreement, not independent ground truth). Official decoding, 2 samples per question. Promotion
rule fixed in advance: an earlier checkpoint is promoted to the full 8-sample evaluation only if its pass@2 beats the final
checkpoint by more than 0.03 with a lower cap rate.

| checkpoint | acc per sample | pass@2 | samples at 32K cap | mean tokens |
|---|---:|---:|---:|---:|
| selected, epoch 1 | 0.153 | 0.240 | 55.5% | 22.9K |
| selected, epoch 2 | 0.250 | 0.375 | 28.3% | 17.0K |
| selected, epoch 3 (final) | 0.290 | 0.440 | 23.5% | 16.4K |
| random, epoch 1 | 0.183 | 0.285 | 52.8% | 22.0K |
| random, epoch 2 | 0.213 | 0.340 | 32.0% | 17.6K |
| random, epoch 3 (final) | 0.263 | 0.380 | 21.8% | 16.1K |

The panel is hard (teacher traces up to 75K characters) and over half of the epoch-1 samples hit the cap, so it mostly measures
whether a checkpoint finishes its reasoning. Both arms improve monotonically with training on every column; the final
checkpoints are the best ones, so no earlier checkpoint was promoted and the main comparison stands as the 3-epoch models.
The screening also shows that more SFT reduces, rather than causes, the run-on behaviour: the cap rate falls from ~55% to
~24% between epoch 1 and epoch 3. Files: `eval_outputs/dev_screen/`, panel `data/dev200.jsonl`.

## 6c. A student-dependent selector: learning progress (third data arm)

The first selector describes the data (category, duplicates, length), not what Qwen learns from it. The third arm asks a
question about the student instead: *how much does Qwen's fit to this solution improve after a little related training?*

**Method (rule frozen before any result of the resulting model was seen).**

1. Eligible pool: math rows whose full conversation fits 16,384 tokens, with non-empty output and balanced `<think>` tags; the
   200 dev-panel rows are excluded so the panel stays a clean check (64,719 rows). Duplicated prompts are kept at this stage.
2. Probe SFT: 2,000 random eligible rows, 1 epoch, otherwise the main recipe (250 steps, 47 min on one A100).
3. Candidates: the other 62,218 eligible rows (55,414 distinct prompts); rows sharing a prompt with a probe row are excluded so
   memorised probe answers cannot inflate scores.
4. Score: teacher-forced mean response-token cross-entropy under base Qwen minus the same under the probe model
   (`scripts/score_loss.py`; no generation needed, about 1.5 A100-hours per 15K rows per model).
5. One response per prompt: among rows with the same prompt, keep the response with the largest score (GRAPE-style choice of
   the demonstration that best fits the student).
6. Coverage: same source x token-band strata as the first selected set, quotas proportional to distinct prompts per stratum;
   top-score rows within each stratum (`scripts/select_lp.py`).
7. Train from the original Qwen checkpoint with the unchanged recipe (3 epochs, global batch 8, lr 5e-6), then the official
   evaluation and the dev-panel screen.

**What the selector picked.** 15,000 rows, 99.1M training tokens per epoch (random 113.4M, first selected set 107.6M);
OpenMathReasoning 11,891 / NuminaMath-CoT 3,109; median loss drop 0.31 versus 0.27 over all candidates. Caveat recorded before
training: selected rows have a lower base loss on average (1.00 versus 1.60 for all candidates), so the signal leans toward
solutions Qwen already finds predictable; the stratification keeps long traces in, but the lean is real. Overlap: 3,753 rows
with the first selected set, 2,241 with random.

**Results (same 3-epoch recipe as the other arms; `reports/five_models/results.md`).**

| | cn_math | kaoyan | amc | minerva | olympiad | gpqa | **macro pass@8** | **macro pass@1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.367 | 0.548 | 0.650 | 0.452 | 0.498 | 0.717 | 0.539 | 0.262 |
| random 15K | 0.333 | 0.653 | 0.575 | 0.471 | 0.526 | 0.737 | 0.549 | 0.246 |
| selected 15K | 0.300 | 0.688 | 0.700 | 0.463 | 0.520 | 0.722 | 0.566 | 0.244 |
| learning progress 15K | 0.400 | 0.673 | 0.650 | 0.463 | 0.526 | 0.753 | **0.578** | **0.265** |

(per-benchmark cells are pass@8). Paired bootstrap on the macro (B = 2000, 95% CI): random -> learning progress is +0.028
[-0.011, +0.069] pass@8 and **+0.019 [+0.005, +0.032] pass@1**; first selected set -> learning progress is +0.012 [-0.025,
+0.050] pass@8 and **+0.021 [+0.008, +0.034] pass@1**; base -> learning progress is +0.039 [-0.004, +0.082] pass@8 and
+0.003 [-0.017, +0.023] pass@1. It is the best of the equal-budget arms on both metrics, with fewer training tokens than
either (99.1M per epoch versus 113.4M and 107.6M), and the only 3-epoch model that does not lose pass@1 relative to the base
model. The pass@1 gain over both other arms is outside the noise; the pass@8 gain is not.

Where it comes from: fewer run-on answers. Its cap rate on the six benchmarks is 4.3% (kaoyan) to 17.4% (GPQA), versus
7.7-23.4% for the first selected set and 8.7-26.5% for random, and the held-out panel shows the same (capped 14.8% versus 23.5%
and 21.8%; acc 0.278, pass@2 0.405, between the two other arms). The lean recorded before training shows on the hardest set:
on AIME24 (not in the macro) it is the weakest fine-tuned model, 0.200 pass@8 versus 0.333 for random (difference CI [-0.267,
-0.033]), consistent with a selector that prefers solutions the student already finds predictable.

**Training longer (selected arm, epochs 4-5).** Separate optimisation run, resumed from the epoch-3 optimizer state with a
warm-restarted cosine (1.7e-6 to 0 over 3,750 steps); final train loss 0.19 versus 0.53 at epoch 3. Dev panel: acc 0.313,
pass@2 0.420, capped 15.5%, mean 14.5K tokens (epoch 3: 0.290 / 0.440 / 23.5% / 16.4K).

Official evaluation of the epoch-5 model (all 8 benchmarks, same protocol; `reports/five_models/results.md`):

| | macro pass@8 | macro pass@1 | cap rate on the 6 benchmarks |
|---|---:|---:|---|
| base | 0.539 | 0.262 | 0.1-2.5% |
| random 15K (3 epochs) | 0.549 | 0.246 | 8.7-26.5% |
| selected 15K (3 epochs) | 0.566 | 0.244 | 7.7-23.4% |
| selected 15K, 5 epochs | **0.590** | **0.269** | 4.8-16.7% |

Paired bootstrap (B = 2000, 95% CI) on the macro: epoch 3 -> epoch 5 is +0.024 [-0.013, +0.064] pass@8 and
**+0.025 [+0.010, +0.039] pass@1**; random -> epoch 5 is +0.040 [-0.004, +0.086] pass@8 and **+0.023 [+0.009, +0.038]**
pass@1; base -> epoch 5 is **+0.051 [+0.009, +0.095]** pass@8 and +0.007 [-0.011, +0.027] pass@1. Per benchmark, epoch 5
versus epoch 3: AMC +3 of 40 questions at pass@8, GPQA +7 of 198, OlympiadBench +0.018, kaoyan and Minerva -2 questions each,
CN-Math +1; pass@1 rises on all six (significantly on Minerva, OlympiadBench and GPQA). The mechanism is the one 5.4 points
to: the share of samples cut off at 32,768 tokens falls on every benchmark (OlympiadBench 20.9% -> 15.4%, GPQA 23.4% -> 13.9%,
Minerva 16.2% -> 11.9%, kaoyan 7.7% -> 4.8%). A capped sample is almost always wrong (accuracy 0 to 3%), while the accuracy
of samples that do finish is about unchanged (OlympiadBench 0.355 vs 0.362, GPQA 0.298 vs 0.291), so fewer cut-offs translate
directly into pass@1.

How to read this: it is the first fine-tuned model that beats the base model on pass@8 with a confidence interval excluding
zero, and the first that recovers the base model's pass@1. It is **not** an equal-budget data comparison: the random arm was
not trained to 5 epochs, and the decision to train longer was taken after the 3-epoch benchmark results were known (motivated
by the held-out panel, which was still improving at epoch 3). It says that, for this 3B student, the training budget matters as
much as the data choice, and that 3 epochs under-train it on reasoning traces this long.

## 6d. Short traces for a small student (fourth data arm)

The capped-sample analysis (5.4) shows where the fine-tuned models lose points: 15 to 27% of samples on Minerva, GPQA and
OlympiadBench never finish thinking within 32,768 tokens. Small students also learn long teacher traces less well than
shorter ones (Li et al., 2025, *Small Models Struggle to Learn from Strong Reasoners*). The fourth arm changes only the length
of the demonstrations.

**Method (rule frozen before any result of the resulting model was seen; `scripts/select_short.py`).** Same candidate pool as
the learning-progress arm (62,218 rows, 55,414 prompts). One response per prompt, the shortest; keep prompts whose full
conversation is at most 8,192 tokens (35,266 prompts qualify); strata = source x token band (up to 2K, 2-4K, 4-8K) with quotas
proportional to distinct prompts, as for the other selected sets; uniform random rows within each stratum (seed 20261018).
Training and evaluation are unchanged (3 epochs, global batch 8, lr 5e-6; official decoding, 32,768-token cap).

**What it picked.** 15,000 rows, 67.0M training tokens per epoch (random 113.4M, first selected set 107.6M, learning progress
99.1M); median 4.4K tokens per example; OpenMathReasoning 11,591 / NuminaMath-CoT 3,409; overlap 2,267 rows with random,
3,777 with the first selected set and 3,795 with the learning-progress set.

**Status (Oct 18, 21:30 ET).** The selection is done and frozen (`data/short15k_manifest.json`). Training and the full
evaluation run as one Tinkercliffs job (`slurm/pipeline_tc.slurm`: environment check, training with the unchanged recipe,
19 benchmark shards plus the held-out panel on 4 A100s, aggregation), rehearsed end to end on Falcon; it is queued, with the
scheduler's estimated start on Oct 20. Results will be added here if they arrive before the deadline.

## 7. Reproducibility and artifacts

- Models (Hugging Face): `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k`, `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k`
- Datasets (Hugging Face): `yuan-zhou-hf/acereason-random15k`, `yuan-zhou-hf/acereason-advanced15k` (json + manifest + pool stats)
- Code: this repo (`scripts/prep_data.py`, `slurm/*.slurm`, `yamls/*.yaml`, `scripts/eval_summary.py`, `scripts/aggregate_results.py`,
  `scripts/push_to_hub.py`); environment freezes `envs/*_freeze.txt`; per-benchmark logs and summaries under `eval_outputs/`;
  training records under `models/`; W&B project `reasoning-sft`; decision log `docs/progress_log.md`.
- Slurm job ids (Falcon): training 645899 (random), 645900 (selected); base eval 645832/646490; fine-tuned evals 646669,
  646672, 647763-647768, OlympiadBench shards 648090-648097 (L40S) and 648458-648473 (A30); finalize 649016.
- Follow-up arms (6b-6d): selection code `scripts/make_probe_and_candidates.py`, `scripts/score_loss.py`, `scripts/select_lp.py`
  (learning progress) and `scripts/select_short.py` (short traces), launcher `slurm/lp_stage.slurm`, dev panel
  `scripts/make_dev_panel.py` + `scripts/dev_screen.py` + `slurm/screen.slurm`, configs `yamls/probe2k.yaml`,
  `yamls/lp15k.yaml` / `yamls/lp15k_2gpu.yaml` (same optimisation, ZeRO-3 optimizer offload for 2 GPUs), `yamls/advanced15k_ep5.yaml`,
  `yamls/short15k.yaml`; manifests `data/lp_build_manifest.json`, `data/lp15k_manifest.json`, `data/short15k_manifest.json`;
  training records `models/{probe2k,lp15k,advanced15k_ep5}/`; evaluation logs `eval_outputs/{lp15k_v1,advanced15k_ep5_v1,dev_screen}/`;
  five-model tables `reports/five_models/`. Single-job pipeline for the short-trace arm on Tinkercliffs: `slurm/pipeline_tc.slurm`
  (rehearsed end to end on Falcon with `PIPE_TEST=1`). W&B runs: lp15k `eo7vne8d`, epochs 4-5 `d1sdo3ad`.
- Slurm job ids (completed runs; failed attempts and their fixes are in `docs/progress_log.md`): Tinkercliffs LP build 8042926,
  probe training 8043031, base-model scoring 8043151-8043154, probe-model scoring 8042932-8042935, selection 8042936; lp15k training 656357 (Falcon,
  2x L40S); epochs 4-5 8044800 (Tinkercliffs, 4x A100); evaluations 657179-657197 + 657676 (epoch 5) and 657369-657387 +
  657954-657971 (lp15k; A30/L40S twins, one copy of each ran); aggregation 657399, 657400.
