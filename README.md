# Reasoning SFT of Qwen2.5-3B-Instruct with a selected 15K training set (course project, Fall 2026)

Team repo for the [Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM) assignment.
We fine-tune Qwen2.5-3B-Instruct on 15K examples from the 100K AceReason-1.1-SFT subset, once with a random subset (the required
baseline) and once with a cheap, label-free selected subset, and evaluate on the 8 course benchmarks.

**Team (Group 2, Trustworthy Machine Learning, Fall 2026, Virginia Tech):** Hanze Li, Sijie Chen, Yuan Zhou, Yiming Liang.
**Teammates: start with [REVIEW.md](REVIEW.md)** (what to read, what to check, how to suggest changes, what we submit).

**Final model:** the selected 15K set trained for 5 epochs: macro pass@8 **59.0%**, pass@1 **26.9%** (base 53.9% / 26.2%,
random baseline 54.9% / 24.6%).

**Report (submitted):** [`reports/paper/Group2_report.pdf`](reports/paper/Group2_report.pdf), NeurIPS 2025 style, source in `reports/paper/`;
Markdown version [`reports/REPORT.md`](reports/REPORT.md).
**Full report:** [`reports/FULL_REPORT.md`](reports/FULL_REPORT.md) (every experiment, including intermediate and follow-up runs).
**Result tables:** [`reports/five_models/results.md`](reports/five_models/results.md).

## Headline (pass@8 / pass@1, %; macro = the 6 leaderboard benchmarks available locally)

| model | AIME24 | MATH-500 | CN-Math | Kaoyan | AMC | Minerva | OlympiadBench | GPQA | **macro** |
|---|---|---|---|---|---|---|---|---|---|
| base Qwen2.5-3B-Instruct | 16.7 / 7.1 | 84.0 / 65.3 | 36.7 / 10.4 | 54.8 / 21.9 | 65.0 / 39.1 | 45.2 / 28.7 | 49.8 / 27.7 | 71.7 / 29.4 | **53.9** / 26.2 |
| SFT random 15K (baseline, 3 epochs) | 33.3 / 7.1 | 88.4 / 64.7 | 33.3 / 7.9 | 65.3 / 29.2 | 57.5 / 34.4 | 47.1 / 25.5 | 52.6 / 27.8 | 73.7 / 23.0 | **54.9** / 24.6 |
| **SFT selected 15K, 5 epochs (final model)** | 23.3 / 7.5 | 87.8 / 66.0 | 33.3 / 10.4 | 67.8 / 29.8 | 77.5 / 38.1 | 45.6 / 27.3 | 53.8 / 30.1 | 75.8 / 25.8 | **59.0** / 26.9 |

Paired bootstrap over questions (95% CI, percentage points): final minus base **+5.1 [+0.9, +9.5]** pass@8; final minus
random +4.0 [-0.4, +8.6] pass@8 and **+2.3 [+0.9, +3.8]** pass@1. The gain comes mainly from fewer answers truncated at the
32K-token limit (4.8-16.7% of samples on the six benchmarks, versus 8.7-26.5% for random). Budget note: random was trained for
3 epochs and the final model for 5; at 3 epochs the selected set scored 56.6% / 24.4%, +1.6 points over random and within
noise. One training run per arm.

## Models and datasets on the Hub (public)

| artifact | link |
|---|---|
| **Final model** (selected 15K, 5 epochs) | https://huggingface.co/yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-advanced15k_ep5 |
| Random-15K baseline model | https://huggingface.co/yuan-zhou-hf/Qwen2.5-3B-Instruct-SFT-random15k |
| Random 15K dataset | https://huggingface.co/datasets/yuan-zhou-hf/acereason-random15k |
| Selected 15K dataset | https://huggingface.co/datasets/yuan-zhou-hf/acereason-advanced15k |

Each model and dataset card names the course, Group 2 and its members, and links this repository.

## Installation

Two conda environments, one for training and one for evaluation (exact package versions in `envs/train_env_freeze.txt` and
`envs/eval_env_freeze.txt`; the full script is `slurm/setup_env.slurm`, run as a 1-GPU job by `slurm/bootstrap.slurm`):

```bash
git clone https://github.com/reds-lab/Project-Reasoning-SFT-LLM   # course repo (we used commit 638989b); bundles LLaMA-Factory
# training env: LLaMA-Factory (course copy, v0.9.4) + torch 2.8.0 (CUDA 12.8) + FlashAttention 2.8.3 + DeepSpeed 0.16.8
conda create -y -p envs/train_env -c conda-forge --override-channels python=3.10 && conda activate envs/train_env
cd Project-Reasoning-SFT-LLM/LLaMA-Factory && pip install -r requirements.txt && pip install -e ".[torch,metrics]" --no-build-isolation
pip install --force-reinstall --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation && pip install deepspeed==0.16.8 liger-kernel==0.8.4 wandb==0.30.0
# evaluation env: vLLM 0.31.0 + the course grader dependencies
conda create -y -p envs/eval_env -c conda-forge --override-channels python=3.10 && conda activate envs/eval_env
pip install vllm==0.31.0 word2number && pip install --no-deps latex2sympy2==1.9.1
pip install "antlr4-python3-runtime==4.9.3" Pebble timeout-decorator python_dateutil tqdm datasets huggingface_hub "sympy>=1.13.3"
```

**Seeds:** data selection `random.Random(20261015)` (`scripts/prep_data.py`); training seed 42 (the LLaMA-Factory /
Transformers default, not overridden in `yamls/*.yaml`); evaluation seed 0, forwarded to vLLM (`patches/eval_seed.patch`).

## Data selection

`scripts/prep_data.py` (deterministic, seed 20261015) produces both sets from the frozen pool file (sha256 in `data/pool_stats.json`):

- **random15k**: uniform sample of the whole pool, unfiltered (the required baseline).
- **advanced15k** (selected): math only (`extra_info.category`), exact-duplicate prompts removed, complete conversations that fit
  the 16,384-token cutoff under the Qwen chat template (so the final answer and end-of-turn token survive), stratified by
  source x total-token band (<=4K, 4-8K, 8-16K) with proportional quotas. No model scoring, embeddings or answer keys needed.
  Compute is roughly matched: 107.6M vs 113.4M training tokens per epoch.

Each set ships with a manifest (row ids, filter counts, token quantiles). Exact benchmark overlap was checked: 6 pool rows match
a benchmark question verbatim; 1 landed in each 15K set and was kept and disclosed rather than sanitizing one arm only.

## Training

LLaMA-Factory (course-bundled copy, v0.9.4), full-parameter SFT, bf16, DeepSpeed ZeRO-3 (no offload), FlashAttention-2,
Liger kernel (fused linear cross-entropy; needed to fit 16K-token samples on 44 GB L40S GPUs), lr 5e-6 cosine, warmup 0,
global batch 8 (4 GPUs x micro-batch 1 x grad-accum 2), cutoff 16,384, response-only loss (label mask verified).
Random baseline: 3 epochs, `yamls/random15k.yaml`, 4x L40S, 9 h 09 min, 338.8M tokens. Final model: epochs 1-3 with
`yamls/advanced15k.yaml` (4x L40S, 8 h 35 min), then epochs 4-5 resumed from the epoch-3 weights, optimizer and scheduler with
`yamls/advanced15k_ep5.yaml` (4x A100, 4 h 26 min; learning rate decays from 1.7e-6 to 0), 538.2M tokens in total.
Launcher: `slurm/train.slurm`. Loss curves: `models/*/training_loss.png`; full table in report section 4.

## Evaluation

Course `eval/` harness (vLLM generation, SymPy grader) with the official settings: temperature 0.6, top-p 0.95, 8 samples,
max 32,768 tokens, seed 0. Our wrapper `slurm/eval_run.slurm` adds strict failure handling, immutable run directories with
manifests, an 8-responses-per-question completeness check, a watchdog for the harness's hang-at-exit, index-range sharding
(`START`/`END`) for long benchmarks, and a generation-length summary (`scripts/eval_summary.py`); `eval.py` was patched by one
line to forward the seed to vLLM (`patches/eval_seed.patch`). `scripts/aggregate_results.py` merges shards and computes pass@1,
pass@8, the leaderboard-style macro average and a paired question-level bootstrap. Fine-tuned models answer in 7K to 21K tokens
on average and 5 to 37% of samples hit the cap, so each fine-tuned evaluation cost ~50 GPU-hours.

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
# 1b. final model: epochs 4-5 on Tinkercliffs 4x A100, resuming from models/advanced15k/checkpoint-5625 (needs its optimizer state)
RUN=advanced15k_ep5 CONFIG=yamls/advanced15k_ep5.yaml sbatch --job-name=rsft_train_adv_ep5 --time=08:00:00 slurm/train.slurm
# 2. evaluation, 1 GPU per job; fine-tuned models need per-benchmark jobs and index shards for OlympiadBench
MODEL=Qwen/Qwen2.5-3B-Instruct RUN_ID=base_v1 sbatch --partition=a30_normal_q --qos=fal_a30_normal_short --time=08:00:00 slurm/eval_run.slurm
MODEL=$P/models/advanced15k_ep5 RUN_ID=advanced15k_ep5_v1 DATASETS="math aime amc cn_math_2024" sbatch --partition=l40s_normal_q --qos=fal_l40s_normal_short --time=20:00:00 slurm/eval_run.slurm
MODEL=$P/models/advanced15k_ep5 RUN_ID=advanced15k_ep5_v1 DATASETS=olympiadbench START=0 END=85 sbatch --partition=a30_normal_q --qos=fal_a30_normal_short --time=23:00:00 slurm/eval_run.slurm
# 3. summary rows for every result file, merged aggregation with paired bootstrap, stopping-failure inspection
sbatch slurm/finalize_summary.slurm
```

Shard merging rule (`scripts/aggregate_results.py`): each question index is taken from the narrowest result shard that covers
it; question texts are checked against `eval/data/<bench>/test.jsonl` at that index; the files used are listed at the end of
`reports/results.md`. `reports/per_question_correctness.json` holds the eight correctness flags per question so the bootstrap
can be recomputed without the raw generations.

Environment freezes: `envs/train_env_freeze.txt`, `envs/eval_env_freeze.txt`. Commits: course repo `638989b`, LLaMA-Factory
course copy (v0.9.4.dev0). Decision and progress log: `docs/progress_log.md`.
