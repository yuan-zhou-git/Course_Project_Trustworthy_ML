# Reasoning SFT of Qwen2.5-3B-Instruct: random vs. selected 15K (course project, Fall 2026)

Team repo for the [Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM) assignment.
We fine-tune Qwen2.5-3B-Instruct on 15K samples from the 100K AceReason-1.1-SFT subset twice, once with a random subset and
once with a cheap, label-free selected subset, and evaluate all three models on the 8 course benchmarks.

**Report:** [`reports/REPORT.md`](reports/REPORT.md). **Full result tables:** [`reports/results.md`](reports/results.md).

## Headline (pass@8 / pass@1; macro = the 6 leaderboard benchmarks available locally)

| model | aime | math | cn_math | kaoyan | amc | minerva | olympiad | gpqa | **macro** |
|---|---|---|---|---|---|---|---|---|---|
| base Qwen2.5-3B-Instruct | .167/.071 | .840/.653 | .367/.104 | .548/.219 | .650/.391 | .452/.287 | .498/.277 | .717/.294 | **.539**/.262 |
| SFT random 15K | .333/.071 | .884/.647 | .333/.079 | .653/.292 | .575/.344 | .471/.255 | .526/.278 | .737/.230 | **.549**/.246 |
| SFT selected 15K | .267/.087 | .886/.652 | .300/.071 | .688/.290 | .700/.350 | .463/.243 | .520/.288 | .722/.224 | **.566**/.244 |

Paired bootstrap, macro pass@8: selected minus random +0.016 [-0.024, +0.056]; selected minus base +0.027 [-0.020, +0.071].
The selected-minus-random gap is concentrated in five AMC questions. One training run per arm, so the comparison is "higher
observed pass@8 in this run", not "reliably superior". Both SFT models lose pass@1 on Minerva and GPQA because 15-27% of their
samples hit the 32K cap while still reasoning (`reports/stopping_failures.md`).

## Models and datasets on the Hub

| artifact | link |
|---|---|
| Random-15K SFT model | `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k` |
| Selected-15K SFT model | `yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k` |
| Random 15K dataset | `yuan-zhou-hf/acereason-random15k` |
| Selected 15K dataset | `yuan-zhou-hf/acereason-advanced15k` |

## Data selection

`scripts/prep_data.py` (deterministic, seed 20261015) produces both sets from the frozen pool file (sha256 in `data/pool_stats.json`):

- **random15k**: uniform sample of the whole pool, unfiltered (the required baseline).
- **advanced15k** (selected): math only (`extra_info.category`), exact-duplicate prompts removed, complete conversations that fit
  the 16,384-token cutoff under the Qwen chat template (so the final answer and end-of-turn token survive), stratified by
  source x total-token band (<=4K, 4-8K, 8-16K) with proportional quotas. No model scoring, embeddings or answer keys needed.
  Compute is roughly matched: 107.6M vs 113.4M training tokens per epoch.

Each set ships with a manifest (row ids, filter counts, token quantiles). Exact benchmark overlap was checked: 6 pool rows match
a benchmark question verbatim; 1 landed in each 15K set and was kept and disclosed rather than sanitizing one arm only.

## Training (identical for both arms)

LLaMA-Factory (course-bundled copy, v0.9.4), full-parameter SFT, bf16, DeepSpeed ZeRO-3 (no offload), FlashAttention-2,
Liger kernel (fused linear cross-entropy; needed to fit 16K-token samples on 44 GB L40S GPUs), lr 5e-6 cosine, warmup 0,
3 epochs, global batch 8 (4 GPUs x micro-batch 1 x grad-accum 2), cutoff 16,384, response-only loss (label mask verified).
Configs: `yamls/random15k.yaml`, `yamls/advanced15k.yaml`; launcher: `slurm/train.slurm`. Measured: ~10.3K tokens/s on
4x L40S, 9 h 09 min (random, 338.8M tokens) and 8 h 35 min (selected, 322.9M tokens). Loss curves: `models/*/training_loss.png`.

## Evaluation

Course `eval/` harness (vLLM generation, SymPy grader) with the official settings: temperature 0.6, top-p 0.95, 8 samples,
max 32,768 tokens, seed 0. Our wrapper `slurm/eval_run.slurm` adds strict failure handling, immutable run directories with
manifests, an 8-responses-per-question completeness check, a watchdog for the harness's hang-at-exit, index-range sharding
(`START`/`END`) for long benchmarks, and a generation-length summary (`scripts/eval_summary.py`); `eval.py` was patched by one
line to forward the seed to vLLM (`patches/eval_seed.patch`). `scripts/aggregate_results.py` merges shards and computes pass@1,
pass@8, the leaderboard-style macro average and a paired question-level bootstrap. Fine-tuned models answer in 8K to 21K tokens
on average and 8 to 37% of samples hit the cap, so each fine-tuned evaluation cost ~50 GPU-hours.

## Reproduce (exactly as run, ARC Falcon cluster)

The `#SBATCH` headers in `slurm/*.slurm` name the TinkerCliffs A100 partition; every real run overrode partition and QoS on the
command line because the A100/H200 queues were saturated. Paths are hard-coded to
`/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft` (`P`) and `/scratch/yuanzhou1/reasoning_sft` (`S`); edit those two
variables at the top of each script for another account.

```bash
# 0. environment (two conda envs, pinned: torch 2.8.0+cu128, flash-attn 2.8.3 wheel, DeepSpeed 0.16.8, LLaMA-Factory 0.9.4 from the
#    course repo; vLLM 0.31.0 in a separate env) + data prep + base-model smoke test, all in one 1-GPU job
sbatch --partition=a30_normal_q --qos=fal_a30_normal_short --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=05:00:00 slurm/bootstrap.slurm
# 1. training, 4x L40S each (Liger kernel is required on 44 GB cards)
RUN=random15k   CONFIG=yamls/random15k.yaml   sbatch --job-name=rsft_train_random --partition=l40s_normal_q --qos=fal_l40s_normal_short --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=14:00:00 slurm/train.slurm
RUN=advanced15k CONFIG=yamls/advanced15k.yaml sbatch --job-name=rsft_train_adv    --partition=l40s_normal_q --qos=fal_l40s_normal_short --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=14:00:00 slurm/train.slurm
# 2. evaluation, 1 GPU per job; fine-tuned models need per-benchmark jobs and index shards for OlympiadBench
MODEL=Qwen/Qwen2.5-3B-Instruct RUN_ID=base_v1 sbatch --partition=a30_normal_q --qos=fal_a30_normal_short --time=08:00:00 slurm/eval_run.slurm
MODEL=$P/models/advanced15k RUN_ID=advanced15k_v1 DATASETS="math aime amc cn_math_2024" sbatch --partition=l40s_normal_q --qos=fal_l40s_normal_short --time=20:00:00 slurm/eval_run.slurm
MODEL=$P/models/advanced15k RUN_ID=advanced15k_v1 DATASETS=olympiadbench START=0 END=85 sbatch --partition=a30_normal_q --qos=fal_a30_normal_short --time=23:00:00 slurm/eval_run.slurm
# 3. summary rows for every result file, merged aggregation with paired bootstrap, stopping-failure inspection
sbatch slurm/finalize_summary.slurm
```

Shard merging rule (`scripts/aggregate_results.py`): each question index is taken from the narrowest result shard that covers
it; question texts are checked against `eval/data/<bench>/test.jsonl` at that index; the files used are listed at the end of
`reports/results.md`. `reports/per_question_correctness.json` holds the eight correctness flags per question so the bootstrap
can be recomputed without the raw generations.

Environment freezes: `envs/train_env_freeze.txt`, `envs/eval_env_freeze.txt`. Commits: course repo `638989b`, LLaMA-Factory
course copy (v0.9.4.dev0). Decision and progress log: `docs/progress_log.md`.
