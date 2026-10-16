# Reasoning SFT of Qwen2.5-3B-Instruct: random vs. selected 15K (course project, Fall 2026)

Team repo for the [Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM) assignment.
We fine-tune Qwen2.5-3B-Instruct on 15K samples from the 100K AceReason-1.1-SFT subset twice, once with a random subset and
once with a cheap, label-free selected subset, and evaluate all three models on the 8 course benchmarks.

**Report:** [`reports/REPORT.md`](reports/REPORT.md). **Full result tables:** [`reports/results.md`](reports/results.md).

## Headline (pass@8 / pass@1; macro = the 6 leaderboard benchmarks available locally)

| model | aime | math | cn_math | kaoyan | amc | minerva | olympiad | gpqa | **macro** |
|---|---|---|---|---|---|---|---|---|---|
| base Qwen2.5-3B-Instruct | .167/.071 | .840/.653 | .367/.104 | .548/.219 | .650/.391 | .452/.287 | .498/.277 | .717/.294 | **.539**/.262 |
| SFT random 15K | .333/.071 | .884/.647 | .333/.079 | .653/.292 | .575/.344 | .471/.255 | .521/.274 | .737/.230 | **.549**/.246 |
| SFT selected 15K | .267/.087 | .886/.652 | .300/.071 | .688/.290 | .700/.350 | .463/.243 | .521/.288 | .722/.224 | **.566**/.244 |

Paired bootstrap, macro pass@8: selected minus random +0.017 [-0.022, +0.057]; selected minus base +0.027 [-0.019, +0.072].
One training run per arm, so the comparison is "improved in this controlled run", not "reliably superior".

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

## Reproduce

```bash
sbatch slurm/bootstrap.slurm                                  # envs + data prep + eval smoke test (one GPU)
RUN=random15k   CONFIG=yamls/random15k.yaml   sbatch --gres=gpu:4 slurm/train.slurm
RUN=advanced15k CONFIG=yamls/advanced15k.yaml sbatch --gres=gpu:4 slurm/train.slurm
MODEL=<ckpt> RUN_ID=<name> DATASETS="math aime" sbatch slurm/eval_run.slurm
MODEL=<ckpt> RUN_ID=<name> DATASETS=olympiadbench START=0 END=169 sbatch slurm/eval_run.slurm   # shard
sbatch slurm/finalize_summary.slurm                           # missing summary rows + aggregation
python scripts/aggregate_results.py --root eval_outputs --runs base=base_v1 random=random15k_v1 advanced=advanced15k_v1 --out reports
```

Environment freezes: `envs/train_env_freeze.txt`, `envs/eval_env_freeze.txt`. Compute: Virginia Tech ARC (Falcon L40S/A30).
Decision and progress log: `docs/progress_log.md`.
