# Reasoning SFT of Qwen2.5-3B-Instruct with a selected 15K training set

Course project report (ECE, Virginia Tech, Fall 2026). Repo: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML.
Assignment: [Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM). Evaluations run 2026-10-15 to
2026-10-18. The complete record of every run, including intermediate and follow-up experiments, is in
[`FULL_REPORT.md`](FULL_REPORT.md); Appendix A summarises them.

## 1. Summary

We fine-tuned Qwen2.5-3B-Instruct on 15,000 examples from the course's 100K AceReason-1.1-SFT subset and evaluated it on the
8 course benchmarks under the official protocol (8 samples per question, temperature 0.6, top-p 0.95, 32,768 max tokens).
Our final model is trained on a **selected** 15K set (math only, duplicate questions removed, every example fits the training
length, stratified by source and length) for 5 epochs. The required baseline is a **random** 15K set trained for 3 epochs.

| model | macro pass@8 (6 leaderboard benchmarks) | macro pass@1 |
|---|---:|---:|
| Qwen2.5-3B-Instruct (base) | 0.539 | 0.262 |
| SFT, random 15K (baseline) | 0.549 | 0.246 |
| **SFT, selected 15K, 5 epochs (final model)** | **0.590** | **0.269** |

Paired bootstrap over questions (B = 2000, 95% intervals): the final model is **+0.051 [+0.009, +0.095]** pass@8 over the base
model, the only fine-tuned gain whose interval excludes zero, and +0.040 [-0.004, +0.086] pass@8 and **+0.023 [+0.009, +0.038]**
pass@1 over the random baseline. Random fine-tuning alone moves the base model by +0.011 [-0.042, +0.062]. The main mechanism
is fewer unfinished answers: fine-tuned models write long reasoning traces, and the share of samples cut off at the 32,768-token
limit (almost always graded wrong) falls from 8.7-26.5% for the random baseline to 4.8-16.7% for the final model.

**Budget note.** The baseline was trained for 3 epochs and the final model for 5, so the final-vs-random gap mixes data choice
and training length. At the same 3-epoch budget the selected set scored 0.566 / 0.244, +0.016 [-0.024, +0.056] pass@8 over
random, within noise (Appendix A.1). We report one training run per arm.

## 2. Setup

| item | value |
|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct` (Qwen Team, 2024), snapshot `aa8e7253` |
| training pool | `redsgnaoh/acereason11_100k`, the course's 100K subset of AceReason-1.1-SFT (Liu et al., 2025): 100,000 rows, 67,432 math / 32,568 code (sha256 in `data/pool_stats.json`) |
| frameworks | LLaMA-Factory 0.9.4 (Zheng et al., 2024; course copy), DeepSpeed 0.16.8 ZeRO-3 (Rajbhandari et al., 2020), FlashAttention 2.8.3 (Dao, 2024), Liger kernel 0.8.4 (Hsu et al., 2024), torch 2.8.0+cu128; vLLM 0.31.0 for evaluation (Kwon et al., 2023) |
| hardware | Virginia Tech ARC. Training: Falcon 4x NVIDIA L40S 48 GB (both models' first 3 epochs), Tinkercliffs 4x A100 80 GB (final model, epochs 4-5). Evaluation: 1x L40S or A30 per job |
| tracking | W&B project `reasoning-sft`; loss curves also in `models/<run>/training_loss.png` |

## 3. Data selection strategy

**Random 15K (required baseline).** `random.Random(20261015).sample` over the whole pool, no filtering: 113.4M training tokens
per epoch; 556 rows exceed the 16,384-token cutoff and are truncated, so their final answer is lost in training.

**Selected 15K (used by the final model).** The benchmarks are math and science, a third of the pool is code, AceReason repeats
many questions (15,906 exact duplicates), and 3.5% of rows do not fit the cutoff. Filters, in order:

1. `extra_info.category == "math"` (removes 32,568 rows);
2. exact-duplicate questions removed, first occurrence kept (removes 7,730);
3. full conversation at most 16,384 tokens under the Qwen chat template (removes 2,292), so no training answer is truncated;
   no empty outputs or unbalanced `<think>` tags were found;
4. from the 57,410 eligible rows, stratified sampling by source x total-token band (up to 4K, 4-8K, 8-16K) with proportional
   quotas, seed 20261015: NuminaMath-CoT 1,162 / 988 / 957, OpenMathReasoning 3,091 / 4,174 / 4,628.

Result: 107.6M training tokens per epoch (5% fewer than random), median example 6.5K tokens, nothing truncated, 2,252 rows
shared with the random set. The method needs no model inference, embeddings or answer keys and is reproducible from
`scripts/prep_data.py` and the frozen pool file. It follows question filtering and source mixing in OpenThoughts (Guha et al.,
2025), prompt diversity in AceReason-Nemotron 1.1 (Liu et al., 2025), and the finding that a small curated set can match a
larger one (LIMA, Zhou et al., 2023; s1, Muennighoff et al., 2025; LIMO, Ye et al., 2025).

**Benchmark overlap.** 6 pool questions match a benchmark question verbatim (3 MATH, 2 OlympiadBench, 1 AMC); one landed in each
15K set and was kept rather than sanitising one arm only. The effect on any score is at most one question.

## 4. Training and hyperparameters

| setting | random baseline | **final model** |
|---|---|---|
| data | random 15K (113.4M tokens/epoch) | selected 15K (107.6M tokens/epoch) |
| method | full-parameter SFT, response-only loss, `template: qwen` | same |
| sequence cutoff | 16,384 tokens | same |
| optimiser | AdamW (Trainer defaults: betas 0.9/0.999, weight decay 0, gradient clipping 1.0), bf16 | same |
| learning rate | 5e-6, cosine, warmup 0 | 5e-6, cosine, warmup 0 for epochs 1-3; epochs 4-5 resume from the step-5,625 state (weights, optimizer, scheduler) on a 5-epoch cosine, so the rate decays from 1.7e-6 to 0 |
| epochs / steps | 3 / 5,625 | 5 / 9,375 |
| batch | global 8 = 4 GPUs x micro-batch 1 x gradient accumulation 2 | same |
| memory and kernels | DeepSpeed ZeRO-3 (no offload), FlashAttention-2, Liger fused linear cross-entropy | same |
| tokens seen / train loss | 338.8M / 0.581 | 538.2M / 0.530 over epochs 1-3, 0.192 over epochs 4-5 |
| hardware and time | 4x L40S, 9 h 09 min | 4x L40S, 8 h 35 min (epochs 1-3) + 4x A100, 4 h 26 min (epochs 4-5) |
| configs | `yamls/random15k.yaml` | `yamls/advanced15k.yaml`, then `yamls/advanced15k_ep5.yaml` |

The Liger fused cross-entropy was needed to fit 16K-token examples on 44 GB GPUs. The label mask was checked on a decoded example
(prompt tokens masked, labels start at `<think>`).

## 5. Evaluation protocol

Course `eval/` harness: vLLM generation with the system prompt "Please reason step by step, and put your final answer within
\boxed{}", the last `\boxed{}` extracted and graded by SymPy-based equivalence. For each question the model generates 8 answers.
**pass@8** counts a question as solved if any of the 8 is correct (pass@k of Chen et al., 2021, with k equal to the 8 samples);
**pass@1** is the average share of correct answers per question. The **macro** score averages the 6 leaderboard benchmarks
available locally (CN-Math 2024, Kaoyan, AMC, Minerva, OlympiadBench, GPQA) with equal weight; AIME25 is on the course
leaderboard but not available locally, and AIME24 and MATH-500 are reported but excluded, following the course definition.
Differences are reported with 95% intervals from a paired bootstrap that resamples questions and keeps each question's 8 samples
together (Efron & Tibshirani, 1993; Miller, 2024).

Our wrapper around the harness leaves the protocol unchanged and adds strict failure handling, run directories with a manifest,
an 8-answers-per-question completeness check, a watchdog for the harness's hang at exit, the seed forwarded to vLLM (one-line
patch, `patches/eval_seed.patch`), and index-range sharding of long benchmarks, merged by question index with each record's
question text checked against the benchmark file.

## 6. Results

### 6.1 All 8 benchmarks (pass@8 / pass@1)

| benchmark (questions) | base | random 15K | **final model** |
|---|---|---|---|
| AIME24 (30) | 0.167 / 0.071 | 0.333 / 0.071 | 0.233 / 0.075 |
| MATH-500 (500) | 0.840 / 0.653 | 0.884 / 0.647 | 0.878 / 0.659 |
| CN-Math 2024* (30) | 0.367 / 0.104 | 0.333 / 0.079 | 0.333 / 0.104 |
| Kaoyan* (199) | 0.548 / 0.219 | 0.653 / 0.292 | 0.678 / 0.298 |
| AMC* (40) | 0.650 / 0.391 | 0.575 / 0.344 | 0.775 / 0.381 |
| Minerva* (272) | 0.452 / 0.287 | 0.471 / 0.255 | 0.456 / 0.273 |
| OlympiadBench* (675) | 0.498 / 0.277 | 0.526 / 0.278 | 0.538 / 0.301 |
| GPQA* (198) | 0.717 / 0.294 | 0.737 / 0.230 | 0.758 / 0.258 |
| **macro (6 starred)** | **0.539 / 0.262** | **0.549 / 0.246** | **0.590 / 0.269** |

### 6.2 Comparison (paired bootstrap, B = 2000, 95% CI; difference in pass@8 unless noted)

| benchmark | random - base | final - base | final - random |
|---|---|---|---|
| AIME24 | +0.167 [+0.033, +0.333] | +0.067 [-0.067, +0.200] | -0.100 [-0.200, +0.000] |
| MATH-500 | +0.044 [+0.014, +0.072] | +0.038 [+0.008, +0.068] | -0.006 [-0.028, +0.018] |
| CN-Math 2024 | -0.033 [-0.267, +0.233] | -0.033 [-0.233, +0.167] | +0.000 [-0.167, +0.200] |
| Kaoyan | +0.106 [+0.045, +0.166] | +0.131 [+0.070, +0.191] | +0.025 [-0.020, +0.070] |
| AMC | -0.075 [-0.225, +0.075] | +0.125 [+0.000, +0.250] | +0.200 [+0.050, +0.375] |
| Minerva | +0.018 [-0.026, +0.062] | +0.004 [-0.044, +0.051] | -0.015 [-0.055, +0.022] |
| OlympiadBench | +0.028 [-0.006, +0.061] | +0.040 [+0.006, +0.073] | +0.012 [-0.016, +0.039] |
| GPQA | +0.020 [-0.056, +0.096] | +0.040 [-0.030, +0.111] | +0.020 [-0.045, +0.086] |
| **macro pass@8** | **+0.011 [-0.042, +0.062]** | **+0.051 [+0.009, +0.095]** | **+0.040 [-0.004, +0.086]** |
| **macro pass@1** | **-0.016 [-0.036, +0.004]** | **+0.007 [-0.011, +0.027]** | **+0.023 [+0.009, +0.038]** |

Per-benchmark pass@1 and all tables in machine-readable form: `reports/five_models/results.md` and `results.json`;
`per_question_correctness.json` holds the 8 correctness flags per question so the bootstrap can be recomputed.

### 6.3 Answer length and cut-off answers (`reports/five_models/stopping_failures.md`)

| model | mean answer tokens (range over benchmarks) | samples cut off at 32,768 tokens: 6 leaderboard benchmarks / all 8 | accuracy of finished vs cut-off samples | eval GPU-hours |
|---|---|---|---|---:|
| base | 0.7K (Minerva) to 2.7K (AIME) | 0.1-2.5% / 155 of 15,552 (1.0%) | MATH 0.657 vs 0.000 | ~3.5 |
| random 15K | 7.8K (MATH) to 21.1K (AIME); OlympiadBench 14.5K | 8.7-26.5% / 2,643 (17.0%) | MATH 0.707 vs 0.023; Minerva 0.296 vs 0.016; GPQA 0.312 vs 0.000 | ~50 |
| final model | 7.1K (MATH) to 21.3K (AIME); OlympiadBench 12.9K | 4.8-16.7% / 1,809 (11.6%) | MATH 0.702 vs 0.031; Minerva 0.309 vs 0.000; GPQA 0.298 vs 0.005 | ~54 |

Almost all cut-off samples of the fine-tuned models are unfinished reasoning (the `</think>` tag is never reached, 86-100% by
benchmark); a minority end in a repeating loop.

## 7. Analysis

- **What fine-tuning changed.** Both fine-tuned models write DeepSeek-R1-style `<think>` traces 10 to 20 times longer than the
  base model's answers, the over-long reasoning reported for o1-like models (Chen et al., 2024). pass@8 rises with clear
  intervals on Kaoyan and MATH-500 for both. Random fine-tuning lowers pass@1 on Minerva (-0.033 [-0.058, -0.008]) and GPQA
  (-0.064 [-0.102, -0.026]), the benchmarks where 15-27% of its samples are cut off; a cut-off sample has no final answer and is
  graded wrong. On finished samples its accuracy matches or beats the base model, so the loss is a stopping problem rather than a
  reasoning one.
- **Why the final model is better.** It cuts the share of unfinished answers on every leaderboard benchmark (for example GPQA
  26.5% -> 13.9%, OlympiadBench 22.4% -> 15.4%, Minerva 14.8% -> 11.9% versus random), while the accuracy of finished samples
  stays about the same, so fewer cut-offs translate directly into pass@1 (+0.023 macro, clearest on OlympiadBench, +0.023
  [+0.012, +0.035]) and into pass@8, led by AMC (+8 of 40 questions). Two choices are likely to contribute. The selected data
  keeps every example within the training length, so the model always sees complete solutions that end with an answer (random
  has 556 truncated examples). And longer training helps: on a held-out panel of 200 pool questions in neither training set, the selected run's cut-off rate falls from 55% after
  epoch 1 to 24% after epoch 3 and 16% after epoch 5 (Appendix A.2). The AceReason-Nemotron 1.1 study of the same data family
  likewise finds SFT accuracy "gradually improves from the 1st to the 5th epoch" before plateauing around epochs 5-6 (Liu et al.,
  2025).
- **How much is the data choice.** At equal budget (3 epochs) the selected set leads random by +0.016 macro pass@8, inside the
  noise and concentrated in five AMC questions; with a second generation seed the AMC gap shrinks to two questions (Appendix A.1).
  So the final model's lead over the baseline comes from selected data plus longer training together, and this study cannot
  separate the two: random was not trained for 5 epochs.
- **Limitations.** One training run per arm; the bootstrap covers question sampling only, not training-seed or data-draw
  variance. AIME24 and CN-Math have 30 questions each and move by 0.033 per question. The decision to train for 5 epochs was taken
  after the 3-epoch benchmark results were known (motivated by the held-out panel, which was still improving), and the final model
  is chosen on the same local benchmarks it is reported on; AIME25 on the course leaderboard is the only unseen test. All training
  data comes from the course pool: the assignment allows adding outside data, which we did not do, partly because outside math
  sets often contain the benchmark problems themselves and would need decontamination. Natural next steps are short curated
  reasoning sets such as LIMO's or s1's (Ye et al., 2025; Muennighoff et al., 2025), which fit the observation that small
  students learn better from shorter reasoning (Li et al., 2025), and science data for GPQA, for which our math-only selected set
  has no in-domain examples.
- **Against the Fall-2025 leaderboard.** That table averages 7 benchmarks including AIME25, so it is not directly comparable;
  our base model reproduces its baseline row within benchmark noise (MATH 0.840 exactly).

## 8. Reproducibility and artifacts

- **Hugging Face models:** final model `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k_ep5`; random baseline
  `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k`.
- **Hugging Face datasets:** `yuan-zhou-hf/acereason-random15k` and `yuan-zhou-hf/acereason-advanced15k` (the 15K json in
  LLaMA-Factory format, its manifest of pool row ids, and pool statistics).
- **Code (this repo):** data selection `scripts/prep_data.py`; training configs `yamls/random15k.yaml`, `yamls/advanced15k.yaml`,
  `yamls/advanced15k_ep5.yaml`; launchers `slurm/train.slurm`, `slurm/eval_run.slurm`, `slurm/finalize_summary.slurm`;
  aggregation and analysis `scripts/eval_summary.py`, `scripts/aggregate_results.py`, `scripts/inspect_caps.py`; upload
  `scripts/push_to_hub.py`; environment freezes `envs/*_freeze.txt`.
- **Records:** per-benchmark logs, manifests and summaries in `eval_outputs/{base_v1,random15k_v1,advanced15k_ep5_v1}/`; training
  records in `models/{random15k,advanced15k,advanced15k_ep5}/`; decision log `docs/progress_log.md`. The final checkpoints,
  including optimizer states, are kept on ARC project storage.
- **Slurm job ids:** training 645899 (random), 645900 (selected, epochs 1-3, Falcon) and 8044800 (epochs 4-5, Tinkercliffs);
  evaluation of base 645832/646490; random and the 3-epoch selected checkpoint 646669, 646672, 647763-647768 with OlympiadBench
  shards 648090-648097 and 648458-648473 (aggregated by 649016); final model 657179-657197 and 657676 (aggregated by
  657399/657400).

## Appendix A. Additional experiments

Details, tables and caveats for all of these are in [`FULL_REPORT.md`](FULL_REPORT.md).

**A.1 Selected 15K at 3 epochs (equal-budget check).** The first 3 epochs of the final model's run, evaluated in full: macro
0.566 / 0.244 (pass@8 / pass@1), +0.016 [-0.024, +0.056] pass@8 and -0.002 [-0.014, +0.010] pass@1 over random. The pass@8 gap is
concentrated in AMC (five questions); with a second generation seed on AMC (seed 1) it shrinks to two questions, while the
random model alone moved five questions between seeds.

**A.2 Checkpoint screen on a held-out panel.** 200 pool questions in neither training set, stratified by teacher-solution length,
graded against the teacher's final answer, 2 samples each (pass@2):

| checkpoint | selected: acc / pass@2 / cut off | random: acc / pass@2 / cut off |
|---|---|---|
| epoch 1 | 0.153 / 0.240 / 55.5% | 0.183 / 0.285 / 52.8% |
| epoch 2 | 0.250 / 0.375 / 28.3% | 0.213 / 0.340 / 32.0% |
| epoch 3 | 0.290 / 0.440 / 23.5% | 0.263 / 0.380 / 21.8% |
| epoch 5 (final) | 0.313 / 0.420 / 15.5% | (not trained) |

Both runs improve monotonically through epoch 3, so no earlier checkpoint was preferred; after epoch 5, accuracy per sample rises
and cut-offs fall further, with pass@2 flat within noise.

**A.3 Learning-progress selection (3 epochs).** A model-dependent selector: each candidate solution is scored by how much a short
probe fine-tune on 2,000 random pool rows lowers Qwen's loss on it (the reducible loss of RHO-LOSS, Mindermann et al., 2022;
close in spirit to LESS, Xia et al., 2024), keeping the best-fitting response per question as in GRAPE (Zhang et al., 2025),
with the same stratified quotas. Same 3-epoch recipe: macro 0.578 / 0.265; +0.028 [-0.011, +0.069] pass@8 and **+0.019 [+0.005,
+0.032]** pass@1 over random, with fewer training tokens (99.1M per epoch). It is the strongest equal-budget arm, but weaker on
the hardest set (AIME24 0.200 vs 0.333 for random).

**A.4 Short-trace selection.** A fourth set limited to examples of at most 8K tokens was selected, but its training landed on a
Tinkercliffs node with power-throttled GPUs and was stopped at step 2,779 of 5,625 before any evaluation; no results are reported.

## References

- Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Chen, X. et al. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024.
- Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Guha, E. et al. (2025). OpenThoughts: Data Recipes for Reasoning Models. arXiv:2506.04178.
- Hsu, P.-L. et al. (2024). Liger Kernel: Efficient Triton Kernels for LLM Training. arXiv:2410.10989.
- Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.
- Li, Y. et al. (2025). Small Models Struggle to Learn from Strong Reasoners. arXiv:2502.12143.
- Liu, Z. et al. (2025). AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy. arXiv:2506.13284.
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640.
- Mindermann, S. et al. (2022). Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt. ICML 2022.
- Muennighoff, N. et al. (2025). s1: Simple Test-Time Scaling. arXiv:2501.19393.
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- Rajbhandari, S. et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC 2020.
- Xia, M. et al. (2024). LESS: Selecting Influential Data for Targeted Instruction Tuning. ICML 2024.
- Ye, Y. et al. (2025). LIMO: Less is More for Reasoning. arXiv:2502.03387.
- Zhang, D., Dai, Q. and Peng, H. (2025). The Best Instruction-Tuning Data are Those That Fit. NeurIPS 2025; arXiv:2502.04194.
- Zheng, Y. et al. (2024). LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. ACL 2024 (System Demonstrations).
- Zhou, C. et al. (2023). LIMA: Less Is More for Alignment. NeurIPS 2023.
