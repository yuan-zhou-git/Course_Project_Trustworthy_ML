# Reasoning SFT: Execution Plan on ARC Tinkercliffs

**Written:** 2026-10-15 (5 days to the deadline, Tue 2026-10-20 23:59 ET)
**Inputs:** the assignment README, the preliminary-knowledge file, the 2026-10-15 plan audit, the ARC guide.
**Status legend:** [done] [running] [todo] [blocked: needs the user]

---

## 1. Scope decision

The audit's recommendation is adopted: **two clean, comparable training runs** (random 15K and one inexpensive
selected 15K), a trustworthy evaluation of the base model and both checkpoints, and a transparent selection script.
Dropped from the first pass: model-based difficulty scoring, embedding clustering, LIMOPro pruning. A second training
seed or a second selection variant is budgeted only after both required arms and their evaluations are secured.

What the report can claim is governed by the audit's acceptance table: one run per arm supports "improved in this
controlled run", not "reliably superior".

## 2. Where things live on the cluster

| What | Path | Notes |
|---|---|---|
| Project root (persistent, important) | `/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/` | new folder, separate from `bass_repro` and `rdrl_repro` |
| Code | `.../reasoning_sft/code/{Project-Reasoning-SFT-LLM,LLaMA-Factory}` | shallow clones, commits recorded in the report |
| Conda envs | `.../reasoning_sft/envs/{train_env,eval_env}` | two frozen envs, see section 3 |
| HF cache (model weights, datasets) | `.../reasoning_sft/hf_cache` | `HF_HOME` in every job |
| Training pool and selections | `.../reasoning_sft/data/{raw,selections}` | 100K json, random15k, advanced15k, manifests |
| Final checkpoints | `.../reasoning_sft/models/<run>` | exported model + tokenizer + chat template |
| Eval outputs | `.../reasoning_sft/eval_outputs/<run_id>/` | one immutable directory per (model, settings) |
| Job scripts and logs | `.../reasoning_sft/slurm`, `.../reasoning_sft/logs` | logs named `rsft_<job>.<jobid>.out` |
| Fast IO (purged on a timer) | `/scratch/yuanzhou1/reasoning_sft/{data,ckpt,tmp,pip_cache}` | training-data copy, DeepSpeed checkpoint staging, pip cache |
| Local mirror of scripts | `C:\work\wrokspace\work\courses\26 fall\cluster\` | edit locally, scp, strip CRLF, md5-compare |

Rules in force: nothing computes on the login node; every job requests a GPU; job names start with `rsft_`;
the user QoS limit is 10 jobs and the RDRL project holds up to 8 `rdrl_*` jobs, so `rsft_*` stays at 2 to 4 concurrent;
every remote call is wrapped in `timeout` with `BatchMode=yes`.

## 3. Slurm resources, checked 2026-10-15

| Item | Value |
|---|---|
| Account | `tml_2026` (attached to `yuanzhou1`, fair-share 0.06, zero usage so far) |
| Partition | `a100_normal_q` (14 nodes x 8 A100-80GB); `h200_normal_q` (6 nodes, all allocated at check time) |
| QoS | `tc_a100_normal_short`: priority 1200 (base is 1000), max wall 1 day, 10 jobs/user, 42 GPUs/user, usage factor 2 |
| H200 short QoS | 5 jobs/user, so eight parallel single-benchmark H200 jobs are not possible |
| Queue at check time | no pending jobs in either GPU partition; A100 nodes in `mixed` state with free GPUs |
| Modules | `Miniconda3/25.11.1-1`, `CUDA/12.6.0` |

Two conda environments, because LLaMA-Factory pins `transformers<=4.55` and current vLLM needs a newer one:
`train_env` (LLaMA-Factory, FlashAttention-2, DeepSpeed 0.16.8) and `eval_env` (vLLM, grader deps). Both `pip freeze`
outputs are saved under `envs/` and go into the report.

## 4. Work breakdown

### Phase A: environment and data (Oct 15)
- [running] `rsft_setup` (job 7996780): builds both envs on a GPU node, prefetches Qwen2.5-3B-Instruct and the 100K pool.
- [todo] `rsft_prep`: `prep_data.py` audits the pool schema, hashes the file, builds the **random 15K** (fixed seed,
  original row ids), tokenizes every full conversation with the Qwen chat template, dedups prompts, checks exact
  overlap with the 8 benchmarks, and builds the **advanced 15K** (section 5). Writes manifests and stats.
- [todo] register both datasets in `LLaMA-Factory/data/dataset_info.json`; copy the two jsons to scratch for training IO.
- [blocked: needs the user] `huggingface-cli login` and `wandb login` on the cluster. Without them: GPQA evaluation
  may fail (gated dataset) and nothing can be pushed to the Hub. Training logs fall back to `report_to: none`
  until W&B is logged in.

### Phase B: pilots (Oct 15 to 16)
- [todo] `rsft_evalpilot`: eval_env, 1 A100, the `amc` benchmark (40 questions x 8 samples) with the official settings.
  Records startup time, tokens/s, mean and p95 generated length, fraction hitting the cap, and confirms how the pinned
  vLLM handles `max_tokens=32768` against the model's 32,768 context.
- [todo] `rsft_trainpilot`: train_env, 4 A100, ZeRO-3 without offload, `max_steps=40`, a 400-sample slice that
  includes the longest eligible examples, one real checkpoint save. Records peak memory and tokens/s.
  If 4 GPUs are not scheduled within ~1 h, resubmit as 1 A100 no-offload fit test, then offload.
- Decision rule after the pilots: `train_hours = 3 x encoded_tokens / tokens_per_s / 3600 x 1.3`. If the full
  3-epoch run exceeds 20 h on the available GPUs, cut to a common shorter budget for **both** arms before any
  benchmark is seen, and document it.

### Phase C: baseline and random arm (Oct 15 to 16)
- [todo] `rsft_eval_base`: full 8-benchmark evaluation of the base model, 1 A100, resumable, into an immutable run dir.
- [todo] `rsft_train_random`: full SFT, settings from section 6, `--time` from the pilot.
- [todo] `rsft_eval_random`: the same evaluation on the exported random-arm checkpoint. Split into 2 to 4 single-benchmark
  jobs only if the long-pole benchmark (olympiadbench) would exceed the 1-day wall.

### Phase D: advanced arm (Oct 16 to 17)
- [todo] `rsft_train_adv` with identical hyperparameters, starting from the original model.
- [todo] `rsft_eval_adv`.

### Phase E: freeze and deliver (Oct 17 to 19)
- [todo] completeness checks: every question has 8 responses, counts match benchmark sizes, no stale cache reuse.
- [todo] paired bootstrap over question ids per benchmark for the macro average.
- [todo] export: model + tokenizer + chat template, reload test with a short generation.
- [todo] push model and both datasets to the Hub (needs the user's token); verify links in a fresh session.
- [todo] report and GitHub README; `pip freeze`, commits, job ids, measured runtimes, all per-benchmark pass@1/pass@8.
- Oct 19 is buffer. Oct 20: final link verification and submission.

## 5. Selection method for the advanced 15K (frozen before any benchmark is seen)

Inexpensive, label-free, reproducible from one script and the frozen pool file.

1. Freeze: pool file sha256, dataset revision, original row ids.
2. Random 15K: `random.Random(seed).sample` over the whole pool. Kept as-is (no filtering) because it is the required baseline.
3. Dedup prompts by normalized text hash; report exact benchmark overlap for both arms (report-only unless nonzero,
   then disclosed and handled identically in both arms).
4. Keep math by `extra_info.category` (the field name is verified by the script before use; it aborts otherwise).
5. Exact encoded length: full conversation through the Qwen chat template (system, user, assistant, terminal tokens).
   Keep examples whose total length is at most 16,384 tokens so the final answer and end-of-turn token survive training.
6. Reject clear corruption only: empty output, unbalanced `<think>` tags. Proof questions without `\boxed{}` are kept.
7. Stratify by source x length band (up to 4K, 4K to 8K, 8K to 16K total tokens); proportional quotas by
   largest remainder; fixed seed; sampling without replacement; deterministic redistribution of undersupplied strata.
   If fewer than 15K eligible unique examples remain, stop and report.
8. Audit trail: selected ids, filter counts, source and category mix, token quantiles, total encoded and supervised tokens.

## 6. Training settings (identical for both arms)

| Setting | Value |
|---|---|
| model | `Qwen/Qwen2.5-3B-Instruct`, same revision for both arms (snapshot in `hf_cache`) |
| method | full-parameter SFT, bf16, `template: qwen`, FlashAttention-2, gradient checkpointing on |
| DeepSpeed | `ds_z3_config.json` (no offload) on 4 GPUs; offload only as a measured fallback |
| global batch | 8 = 4 GPUs x micro-batch 1 x grad-accum 2 (log world size and step count) |
| LR, schedule, epochs | 5e-6, cosine, warmup 0, 3 epochs (or a common shorter budget decided after the pilot) |
| cutoff_len | 16,384 |
| checkpoints | `save_strategy: epoch`, `save_total_limit: 2`, staged on scratch, final export copied to `models/` |
| logging | `logging_steps: 1`; W&B when logged in, otherwise none |
| launcher | one Slurm task, `FORCE_TORCHRUN=1 llamafactory-cli train`, no outer `srun` |

Before the long job: decode one processed example and check the label mask covers only the response.

## 7. Evaluation protocol and repairs

Official settings are kept: temperature 0.6, top-p 0.95, 8 samples, max 32,768 tokens, seed 0, the course system prompt.
Repairs applied in a wrapper around the course `eval.py` (the Python is patched only to forward `seed` to vLLM):

- `set -euo pipefail` and an explicit exit code check after the `tee` pipeline.
- One immutable output directory per (model, settings) with a manifest (model path and hash, env freeze, git commits,
  settings, job id). Never evaluate a changed checkpoint under a path that already has cached results.
- Completeness check after each benchmark: question count equals the benchmark size and every record has 8 responses.
- Record effective max output length and finish reasons from the saved completions.

Report pass@1, pass@8, mean generated length and GPU-hours for every benchmark; AIME24 and MATH are reported but
excluded from the leaderboard-style macro average, and AIME25 is not available locally.

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| 4-GPU allocation waits too long | 1 A100 no-offload fit test; then offload; then `h200_normal_q` single GPU |
| flash-attn wheel missing for the installed torch | compile inside the setup job (`MAX_JOBS=8`), or pin torch to a version with a wheel |
| vLLM rejects `max_tokens=32768` with the 32K context | record the effective limit; ask the course staff before changing protocol |
| Fine-tuned model evaluation exceeds 1 day on one GPU | split by benchmark into 2 to 4 jobs, same run dir, merge |
| No HF token | evaluation and training proceed on public data; GPQA and Hub pushes wait for the user |
| Scratch purge during a long job | checkpoints written to `$SLURM_TMPDIR`/scratch and copied to `models/` at each save |

## 9. What the user must do

1. On the cluster: `huggingface-cli login` (read/write token, after requesting GPQA access) and `wandb login`.
   These are credentials and are entered by the user, not by the automated session.
2. Confirm the team's GitHub repo and Hugging Face namespace for the final pushes.
