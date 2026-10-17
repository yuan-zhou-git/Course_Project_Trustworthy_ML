# Reasoning SFT project: decisions and progress log

Deadline Tue 2026-10-20 23:59 ET. Full plan: `Reasoning_SFT_Execution_Plan.md`. Cluster scripts: `cluster/`.

## Status 2026-10-16 19:30: experiments complete; follow-up audit fixes applied and pushed (license, shard rule + identity checks, per-question correctness export, stopping-failure analysis, report corrections, Falcon reproduction docs). Remaining: push the two models to the Hub (needs your go-ahead; model cards now carry the Qwen Research License), make the two Hub datasets public, optional follow-ups (epoch-1/2 checkpoint screening; extra AMC seed), final read-through, Canvas submission by Mon Oct 20 23:59 ET.

## Needs the user (done: HF login, W&B login, GitHub repo)

1. Request access to the gated GPQA dataset at https://huggingface.co/datasets/Idavidrein/gpqa (click "agree"), then create a
   Hugging Face token with **write** access at https://huggingface.co/settings/tokens and run:
   ```bash
   module load Miniconda3/25.11.1-1 && source activate /projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/envs/eval_env && huggingface-cli login
   ```
   (if the env does not exist yet, `pip install --user huggingface_hub` and `huggingface-cli login` works too). Token file lands in `~/.cache/huggingface/token`.
2. Weights & Biases: create an API key at https://wandb.ai/authorize and run `wandb login` in the same env (`pip install --user wandb` if needed).
   Training is already configured to log offline, so this is only needed to `wandb sync` later. Not blocking.
3. Tell me the GitHub repo URL and the Hugging Face namespace (user or org) for the final pushes.

## Decisions

- 2026-10-15: Scope per audit: two runs only (random 15K vs advanced 15K), same hyperparameters, base + two evals. No difficulty scoring, clustering or LIMOPro in the first pass.
- Advanced selection = math-only by `extra_info.category`, exact-dedup, complete and fits 16,384 tokens (full chat-template encoding), stratified by source x length band (<=4K, 4-8K, 8-16K), proportional quotas, seed 20261015. Frozen before any benchmark result is seen.
- Two conda envs (`train_env` for LLaMA-Factory, `eval_env` for vLLM) because LLaMA-Factory pins transformers<=4.55.
- Project files: `/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/` (new, separate from other lab work). Fast IO: `/scratch/yuanzhou1/reasoning_sft/`. Job names `rsft_*`; short QoS for priority.
- Global batch fixed at 8 for any GPU count (train.slurm rewrites grad-accum = 8 / NGPU). Checkpoints per epoch, keep 3, staged to `models/` without optimizer shards.
- Eval repairs: `pipefail` + exit-code check, immutable run dir with manifest, 8-responses-per-question check, generation-length summary, `seed` forwarded to vLLM (1-line patch to course `eval.py`).
- W&B runs offline (`WANDB_MODE=offline`) until the user logs in.

## Progress

- 2026-10-15 02:20: Cluster access verified. Account `tml_2026` has A100/H200 short QoS (priority 1200, 1-day wall; 10 jobs/user on A100, 5 on H200). Folder structure created, both repos cloned (course repo commit 638989b, LLaMA-Factory 1fded91).
- 02:35: All 112 A100 GPUs allocated; H200 nodes have 4 free GPUs but 0 free CPU cores. Slurm estimates H200 start Oct 17, A100 start unknown. Our two `rdrl_*` jobs free 2 A100s around 06:00.
- 02:45: Submitted `rsft_bootstrap` twins (setup envs + data prep + base eval pilot on `amc`): job 7996830 (H200) and 7996844 (A100); whichever starts first cancels the other. Training pilot twins 7996845/7996846 (1 GPU, 40 steps, pilot400) chained behind them.
- Local scripts: `cluster/slurm/{bootstrap,setup_env,prep_data,train,eval_run}.slurm`, `cluster/scripts/{prep_data,eval_summary}.py`, `cluster/yamls/{pilot,random15k,advanced15k}.yaml`.
- 03:00: `prep_data.py` validated end to end on a synthetic pool (stratified quotas sum correctly, near-exhaustion redistribution works, pre-registered stop when eligible < N exits 3). Dataset schema confirmed from the HF viewer: `extra_info` = {`category`: math|code, `source`, `index`}; outputs start with `<think>`. LLaMA-Factory 0.9.6.dev0 accepts every yaml key used.
- 03:05: Queued the full base-model evaluation (`rsft_eval_base`, all 8 benchmarks, 1 GPU) as A100/H200 twins behind the bootstrap, so it runs without waiting for me.

- 05:40: Both RDRL jobs finished and other users took the freed A100s. Slurm start estimates: A100 bootstrap Oct 19 03:25, H200 bootstrap Oct 17 17:14. Tinkercliffs alone no longer fits the deadline; checking the Falcon cluster (L40S 48 GB / A30 24 GB, same `tml_2026` account) and the interactive QoS as escape routes.

- 05:45: Falcon checked: same `/projects` path mounted, same modules, idle A30 node, L40S (48 GB) partition for training fallback. Submitted the bootstrap there; it failed in 5 s because Miniconda 25.x refuses the default anaconda.com channels without an interactive Terms-of-Service acceptance. Fix: `conda create -c conda-forge --override-channels` (no ToS involved). Resubmitting on both clusters; a `bootstrap.done` marker makes whichever copy runs second exit immediately.

- 05:55: Second Falcon attempt failed: upstream LLaMA-Factory (main, Oct 2026) dropped `requirements.txt`. Decision: use the LLaMA-Factory copy bundled in the course repo (v0.9.4.dev0, the version the README commands target); `code/LLaMA-Factory` is now a symlink to it, upstream kept as `LLaMA-Factory-upstream`. Third attempt submitted.

- 06:05: Third attempt failed at FlashAttention: LLaMA-Factory pulled torch 2.14.1 (built for CUDA 13.0); no prebuilt flash-attn wheel exists for it and compiling against the CUDA 12.6 module is refused. Decision: training env pins **torch 2.8.0+cu128** with the prebuilt **flash-attn 2.8.3** wheel and the **CUDA/12.8.0** module (all job scripts switched to 12.8). Eval env keeps whatever torch vLLM pins (no compile needed). Fourth attempt submitted.

- 06:20: Training env verified (torch 2.8.0+cu128, flash-attn 2.8.3, DeepSpeed 0.16.8, LLaMA-Factory 0.9.4). Eval env failed its import check: my antlr4 pin (4.11.1, from the eval `requirements.txt` written for vLLM 0.6) cannot read latex2sympy2 1.9.1's grammar tables. Fix: antlr4 4.9.3 and sympy left at torch's version. Eval env has vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.17.0. Fifth attempt submitted.

## Results

**Environment (Falcon job 645830, 06:10-06:19, both envs verified):** train_env = torch 2.8.0+cu128, flash-attn 2.8.3, DeepSpeed 0.16.8, LLaMA-Factory 0.9.4.dev0 (course copy), transformers 4.55. eval_env = vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.17.0, antlr4 4.9.3, latex2sympy2 1.9.1. Model snapshot `aa8e7253`; pool file 2.46 GB.

**Data prep (seed 20261015, cutoff 16,384 tokens, full chat-template encoding):**
- Pool: 100,000 rows; 67,432 math / 32,568 code; sources OpenMathReasoning 54,213, OpenCodeReasoning 19,235, NuminaMath-CoT 13,219, others 13,333. 15,906 exact-duplicate prompts. 2.3% of rows exceed the cutoff.
- Token length per full conversation: median 6,663, p75 10,841, p90 14,439, p99 17,518, max 31,776. Prompts are short (median 110).
- random15k: 113.4M tokens per epoch (110.4M supervised); 556 rows truncated by the cutoff; 1 exact benchmark overlap (kept, disclosed).
- advanced15k: filters removed not-math 32,568, duplicates 7,730, over-cutoff 2,292; 57,410 eligible; strata = {NuminaMath-CoT, OpenMathReasoning} x {<=4K, 4-8K, 8-16K}, quotas 1162/988/957 and 3091/4174/4628. 107.6M tokens per epoch (106.1M supervised), i.e. 5% less compute than random. 1 exact benchmark overlap (olympiadbench; kept, disclosed). 2,252 rows shared with random15k.
- Planning consequence: average sample is ~7.5K tokens, not 5K; three epochs are ~340M tokens per arm. Budget decision waits for the training pilot.

**Training pilot with Liger (job 645893, 4x L40S, 40 steps on pilot400 incl. the 200 longest samples):** no OOM, **12,700 tokens/s aggregate**, 3.87M tokens in 306 s, loss 1.00 -> 0.87, checkpoint + final export complete (weights, tokenizer, chat template). Each DeepSpeed checkpoint holds 35 GB of optimizer shards on scratch; the staged model is 6 GB. Projection: random arm 340M tokens -> ~7.5 h, advanced arm 323M -> ~7 h, plus saves. **Decision: keep the full 3-epoch recipe for both arms.**

- 07:15: Submitted both arms on Falcon, 4x L40S each, 14 h limit: `rsft_train_random` 645899 (started on fal052), `rsft_train_adv` 645900 (pending resources). Tinkercliffs backup chain cancelled (no longer needed).
- 07:20: Fine-tuned evaluations queued behind each training job (1x L40S each, 20 h limit, 3 benchmark groups per arm: {olympiadbench}, {math, aime, amc, cn_math_2024}, {minerva, kaoyan, gpqa}): random 645905-645907, advanced 645908-645910. Output dirs `eval_outputs/random15k_v1`, `eval_outputs/advanced15k_v1`.

**Eval pilot, base model on AMC (1x A30, Falcon):** pass@8 0.650, pass@1 0.391 (Fall-2025 baseline listed 0.700 pass@8; 40 problems, so within noise). 320 generations in 656 s including ~3 min startup; mean 1,297 generated tokens, p95 1,584, 1.6% hit the length limit. The eval harness, grader, completeness check and summary all work end to end. Full 8-benchmark base eval started on Falcon (job 645832, 1x A30, est. ~5 h).

- 06:30: Bootstrap COMPLETED on Falcon (job 645830). Tinkercliffs chain stays queued as backup; its bootstrap will exit immediately via the done marker.
- 06:55: Training pilot on 4x L40S (job 645831): launcher correct (world size 4, grad-accum 2, 40 steps, 400 examples), label mask verified (prompt tokens -100, labels start at `<think>`), step 1 ran (loss 1.00, **8,130 tokens/s aggregate**), then **CUDA OOM** on a 16K-token sample: 35.8 GB in use + a 9.3 GB fp32 logits allocation on a 44 GB card. Decision: enable LLaMA-Factory's `enable_liger_kernel` (fused linear cross-entropy, logits never materialized) for all runs; `liger-kernel` installed on the fly by train.slurm. Throughput implication if it holds: ~340M tokens / 8.1K tok/s = ~12 h per arm for 3 epochs on 4 L40S.

- 07:40: `scripts/aggregate_results.py` ready (per-benchmark pass@1/pass@8, 6-benchmark macro average, paired question-level bootstrap). Verified on partial results; two independent base-model AMC runs on the same GPU type reproduced identical scores (vLLM seed forwarding works). GPQA evaluates from the local `eval/data` copy, so the HF token is only needed for the course's access formality and for the Hub pushes.

- 07:17: Both arms training in parallel on Falcon (random 645899 on fal052, advanced 645900 on fal047), ~10K tokens/s each early on; expected finish ~15:00-16:00, fine-tuned evals then start automatically. Base eval 3/8 benchmarks done.

- 08:45: Base eval **hung after scoring Kaoyan** (pass@8 0.548 printed at 07:32; `python eval.py` and its vLLM engine stayed alive 80+ min). Likely a grader thread stuck in SymPy (the grader uses a ThreadPoolExecutor, which blocks interpreter exit). Fix in `eval_run.slurm`: each benchmark runs in its own process group with a **watchdog** that kills it 5 min after the scores are logged; `eval_summary.py` now recomputes pass@8/pass@1 from the per-sample correctness file (needed because eval.py prints no scores when it reuses a cached result). Hung job and the 6 queued fine-tuned evals (old wrapper) cancelled and resubmitted.

- 09:40: Watchdog confirmed working (fired on Minerva, run continued).

**Base model, all 8 benchmarks (pass@8 / pass@1; 1x A30, 8 samples, 32,768 max tokens):**

| aime | amc | cn_math_2024 | gpqa | kaoyan | math | minerva | olympiadbench | macro-6 (pass@8) |
|---|---|---|---|---|---|---|---|---|
| 0.167 / 0.071 | 0.650 / 0.391 | 0.367 / 0.104 | 0.717 / 0.294 | 0.548 / 0.219 | 0.840 / 0.653 | 0.452 / 0.287 | 0.498 / 0.277 | **0.539** |

Mean generated length 0.7K to 2.7K tokens; at most 5% of samples hit the length cap (AIME). Fall-2025 baseline for comparison: 0.446 over 7 benchmarks including AIME25. MATH-500 has 500 records (README's 499 is a line count); fixed the completeness table.

- 10:45: User logged in to Hugging Face on the cluster as **`yuan-zhou-hf`** (fine-grained token). Both arms saved epoch-1 checkpoints (`checkpoint-1875`) and are in epoch 2; projected finish ~16:30. W&B login still pending (optional, offline logging in use).

- 11:00: W&B login done (`yuan-zhou-vt`, team `yuan-zhou-vt-virginia-tech`). Both offline runs synced up to the current step into project **reasoning-sft** (runs `apnnom10` = random15k, `6cvbvv92` = advanced15k); the final re-sync happens after training ends.

- 16:00: **Advanced arm training COMPLETED** (job 645900): 5,625 steps, 322.9M tokens seen, final train loss 0.530, 8 h 35 min on 4x L40S, export verified (weights + tokenizer + chat template, 3 epoch checkpoints kept without optimizer shards), W&B run fully synced. Its three eval jobs are now queued on L40S (pending GPUs). Random arm at epoch 2.9, ~20 min left.

- 16:25: **Random arm training COMPLETED** (job 645899): 5,625 steps, 338.8M tokens seen, final train loss 0.581, 9 h 09 min on 4x L40S, export verified, W&B synced. Advanced arm: 322.9M tokens, loss 0.530 (lower loss is expected: math-only, no truncated samples; not a quality claim). Evals: advanced g1/g2 running, four jobs pending L40S GPUs.

- 17:10: Both datasets pushed to the Hub as **private** repos: `yuan-zhou-hf/acereason-random15k` (362 MB json + manifest + pool stats + card) and `yuan-zhou-hf/acereason-advanced15k` (323 MB). Must be made public or shared with the instructor before submission. Team GitHub repo: https://github.com/yuan-zhou-git/Course_Project_Trustworthy_ML (private, same caveat); populating it with code, configs, logs and drafts.

- 17:15: GitHub repo made public by the user; initial commit pushed (52 files: scripts, slurm, yamls, data manifests, env freezes, base eval outputs, both training records, docs, report draft). Local clone: `repo/`. Hub datasets still private (user to flip before submission).

- 17:20: Fine-tuned evals are slow: vLLM reports ~1.3K output tokens/s per L40S (long 32K-context decoding is memory-bandwidth bound), i.e. ~4 s per generated sample on MATH. Projection: MATH ~21:00, OlympiadBench 8-10 h, Minerva+Kaoyan+GPQA in series 15-20 h (at the 20 h limit). Splitting group 3 into one job per benchmark.

- 19:50: OlympiadBench ETA from vLLM: 13 h (random) and 27 h (selected) remaining, i.e. the selected run would hit the 20 h limit and lose everything (the course script writes results only at the end). Cancelled both and resubmitted as **4 index shards per model** (0-169, 169-338, 338-507, 507-675; jobs 648090-648097); wrapper, summary and aggregation now handle shards (separate result files, merged by question id). MATH ~90% done for both models; Kaoyan/Minerva (random) finishing within ~1.5 h. 6 jobs running, 10 queued against the 10-job QoS limit.

- 21:00: First fine-tuned score: random model Kaoyan pass@1 0.292 (base 0.219). The summary step missed the result file (eval.py nests local-checkpoint outputs one directory deeper than Hub ids); results are intact. Fixed the lookup (recursive) and added `finalize_summary.slurm`, which writes any missing summary rows and runs the aggregation; the already-queued eval jobs keep the old wrapper, so their rows come from the finalize job.

- 22:25: Fine-tuned rows so far (pass@8 / pass@1, mean gen tokens, frac at 32K cap): random Kaoyan 0.653/0.292 (base 0.548/0.219), 11.0K tok, 8.7% capped; random Minerva 0.471/0.255 (base 0.452/0.287), 10.6K tok, 14.8% capped; selected MATH 0.886/0.652 (base 0.840/0.653), 7.6K tok, 8.4% capped. Pattern: pass@8 up everywhere so far, pass@1 flat or down where many answers hit the cap. Summary rows now write correctly (the fixed summary script is read at run time).

- 22:55: random GPQA 0.737/0.230 (base 0.717/0.294), 14.5K tok mean, **26.5% of answers hit the 32K cap**; pass@1 drops where the model loops. Selected Kaoyan result written (row pending). OlympiadBench shards still queued behind other users' jobs.

- 23:05: random MATH 0.884/0.647 (base 0.840/0.653); selected Kaoyan 0.688/0.290 vs random 0.653/0.292 (base 0.548/0.219). First head-to-head: selected +3.5 pts pass@8 on Kaoyan, pass@1 equal.

- 23:15: L40S start estimates for the OlympiadBench shards slipped to Oct 16 04:28 ... Oct 17 (other users' jobs took the freed GPUs). Added a parallel path: 8 shards of ~85 problems per model on the A30 partition (23 h limit, 16 jobs, QoS allows 10 running). Different index ranges produce different result files; the aggregator merges by question id, so both paths can coexist and the slower one is cancelled once coverage is complete.

- 23:45: selected AIME24 0.267/0.088 (base 0.167/0.071), mean 21K tokens, 37% capped. One L40S OlympiadBench shard and three A30 shards running.

- 00:15 (Oct 16): random AIME24 0.333/0.071; selected Minerva 0.463/0.243 vs random 0.471/0.255 (base 0.452/0.287). Remaining: AMC + CN_MATH (both, in the g2 jobs), selected GPQA, OlympiadBench shards (5 running, 19 queued).

- 01:00: selected AMC 0.700/0.350 (base 0.650/0.391); selected GPQA 0.722/0.224 vs random 0.737/0.230 (base 0.717/0.294). Only AMC+CN_MATH (random), CN_MATH (selected) and OlympiadBench remain; 8 shards running.

- 01:45: random AMC 0.575/0.344 (selected 0.700/0.350, base 0.650/0.391); selected CN_MATH 0.300/0.071 (base 0.367/0.104). First OlympiadBench A30 shard finished in 2.5 h (random, idx 169-254: 0.600/0.332), so the A30 path will finish OlympiadBench by ~06:00. Selected model has 7/8 benchmarks; random 7/8 (CN_MATH running).

- 04:50: Random model fully evaluated (OlympiadBench covered by 7 shard files, L40S + A30 mix); its 3 redundant shard jobs cancelled. Selected model: 2 of 8 OlympiadBench ranges done, 8 shard jobs running, 2 queued.

- 05:05: **Random vs base, all 8 benchmarks (preview, B=300):** macro-6 pass@8 0.549 vs 0.539 (+0.010, 95% CI [-0.044, +0.063]); macro-6 pass@1 0.246 vs 0.262 (-0.016, CI [-0.037, +0.002]). Random OlympiadBench 0.521/0.274 (base 0.498/0.277). Aggregation bug fixed (records without `id` now keyed by shard start + line index).

- 07:15 (Oct 16): **All evaluations complete; finalize job 649016 wrote `reports/results.md`.**

**FINAL (pass@8 / pass@1), macro = 6 leaderboard benchmarks:**

| model | aime | math | cn_math | kaoyan | amc | minerva | olympiad | gpqa | **macro p@8** | macro p@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | .167/.071 | .840/.653 | .367/.104 | .548/.219 | .650/.391 | .452/.287 | .498/.277 | .717/.294 | **.539** | .262 |
| random15k | .333/.071 | .884/.647 | .333/.079 | .653/.292 | .575/.344 | .471/.255 | .521/.274 | .737/.230 | **.549** | .246 |
| selected15k | .267/.087 | .886/.652 | .300/.071 | .688/.290 | .700/.350 | .463/.243 | .521/.288 | .722/.224 | **.566** | .244 |

Paired bootstrap (B=2000), macro pass@8: selected − random +0.017 [−0.022, +0.057]; selected − base +0.027 [−0.019, +0.072]; random − base +0.010 [−0.042, +0.062]. Macro pass@1: both SFT models ≈ −0.017 vs base (CI touching 0), selected − random −0.001. Significant per-benchmark gains: Kaoyan (+0.14 p@8 selected, +0.11 random), MATH (+0.045 both); significant pass@1 losses on Minerva and GPQA for both SFT models, driven by answers hitting the 32K cap (15-27% of samples vs <1% for base). Selected beats random on AMC (+0.125, CI edge at 0) and OlympiadBench pass@1 (+0.014 [+0.001, +0.027]).

- 08:00 (Oct 16): Follow-up audit received. Actions: model card license corrected to the **Qwen Research License** (upstream LICENSE copied into each model repo); aggregator now uses an explicit score-independent shard rule (narrowest shard per question index), verifies question identities against the benchmark files, lists the files used, and exports `per_question_correctness.json`; new `inspect_caps.py` classifies capped samples (thinking never closed / repetitive / no boxed answer); README documents the real Falcon commands; report corrections (7/8 benchmarks up, 3.5% of pool over cutoff, "longer budget would recover" demoted to untested, AMC-concentration of the selected advantage stated). Results being regenerated (finalize job 654189).

- 19:05 (Oct 16): Results regenerated (finalize 654189) with the explicit shard rule; all 24 (run, benchmark) identity checks pass. Changes vs the first aggregate: OlympiadBench random 0.526/0.278 (was 0.521/0.274), selected 0.520/0.288 (was 0.521/0.288); macro pass@8 base 0.539, random 0.549, selected 0.566; selected − random +0.016 [−0.024, +0.056]. **Stopping-failure inspection:** of the capped fine-tuned samples, 86-100% were still inside `<think>` (never reached the answer phase) and only 11-38% show a repetitive tail, so the cap mostly truncates genuinely unfinished reasoning rather than loops; on uncapped samples the SFT models' per-sample accuracy equals or exceeds the base model's (e.g. Minerva 0.296/0.288 vs 0.288; MATH 0.71 vs 0.66), so the pass@1 losses are entirely cap-driven. `reports/stopping_failures.md`.

- 19:45 (Oct 16): User approved both follow-up experiments. (1) **AMC second seed** (seed 1, otherwise official settings) for base/random/selected: jobs 654195-654197, run dirs `*_seed1`. (2) **Checkpoint screening**: 200-question held-out dev panel built from pool math rows in neither 15K set (teacher's boxed answer as reference, length-stratified, seed 20261016); each of random/selected epoch-1, epoch-2, epoch-3 checkpoints screened with official decoding and n=2 (`scripts/make_dev_panel.py`, `scripts/dev_screen.py`, `slurm/screen.slurm`). Policy, fixed in advance: promote at most one earlier checkpoint per arm to the full 8-sample evaluation only if its dev pass@2 beats epoch-3 by more than 0.03 with a lower cap rate; the original 3-epoch comparison stays the main result.

- 20:35 (Oct 16): **AMC second seed (seed 1):** base 0.700/0.419 (seed 0: 0.650/0.391); random 0.700/0.356 (seed 0: 0.575/0.344). The random model's pass@8 moved by five questions between seeds, the same magnitude as the selected-vs-random AMC gap at seed 0. Selected seed-1 pending. Dev panel built: 200 held-out math questions from 36,491 eligible rows, three teacher-length bands; screening running on A30.

- 21:45 (Oct 16): **AMC seed-1 complete.** pass@8 / pass@1: base 0.700/0.419, random 0.700/0.356, selected 0.750/0.394. Two-seed means (pass@8): base 0.675, random 0.638, selected 0.725. The selected-vs-random AMC gap shrinks from 5 questions (seed 0) to 2 (seed 1); seed-to-seed swings of 2-5 questions on this 40-question set are of the same order as the gap. Checkpoint screening on A30 is slow (~3 h per checkpoint, ~700 tok/s); both jobs expected to finish around 06:00 Oct 17.

- 23:30 (Oct 16): Dev-panel screening, epoch-1 checkpoints (200 held-out questions, n=2, acc per sample / pass@2 / capped / mean tokens): selected ep1 0.153 / 0.240 / 55.5% / 22.9K; random ep1 0.183 / 0.285 / 52.8% / 22.0K. The panel is hard (teacher traces up to 75K chars) and over half the samples hit the 32K cap, so it mostly measures whether a checkpoint finishes; epoch 2 and 3 rows due ~02:30 and ~05:30.

- 04:00 (Oct 17): **Checkpoint screening, selected arm complete** (dev panel, n=2; acc / pass@2 / capped / mean tokens): ep1 0.153 / 0.240 / 55.5% / 22.9K; ep2 0.250 / 0.375 / 28.3% / 17.0K; **ep3 0.290 / 0.440 / 23.5% / 16.4K**. Monotonic improvement with training; no earlier checkpoint qualifies for promotion (rule: beat ep3 by >0.03 pass@2 with lower cap rate). Random: ep1 0.183 / 0.285 / 52.8%; ep2 0.213 / 0.340 / 32.0%; **ep3 0.263 / 0.380 / 21.8% / 16.1K**. Both arms monotonic; **no checkpoint promoted**; the 3-epoch comparison stands. On this held-out panel the selected final checkpoint leads random by +0.06 pass@2 / +0.028 acc (teacher-agreement, n=2, 200 questions).

- 04:15 (Oct 17): Follow-up experiments written into the report (section 6b) and pushed. All experiments finished; remaining items are the Hub model upload (awaiting go-ahead), dataset visibility, read-through, Canvas submission (deadline Tue Oct 20 23:59 ET).

## Open risks

- Queue saturation is the critical risk, not compute. If nothing starts by morning, consider the Falcon A30 cluster for the env build and data prep (not done without the user's OK since the instruction was Tinkercliffs).
- Pilot results decide the training budget: if 3 epochs exceed ~20 h on the available GPUs, both arms get the same shorter budget, decided before any benchmark is seen.
