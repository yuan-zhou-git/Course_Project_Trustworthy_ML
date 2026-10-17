#!/bin/bash
#SBATCH --job-name=rsft_eval
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Wrapper around the course eval.py with the audit's repairs: strict failure handling, one immutable run dir
# per (model, settings) with a manifest, per-benchmark completeness check, and a generation-length summary.
# Usage: MODEL=Qwen/Qwen2.5-3B-Instruct RUN_ID=base_v1 DATASETS="amc aime" sbatch --job-name=rsft_eval_base eval_run.slurm
#   MODEL    : HF id or absolute checkpoint path (must contain tokenizer + chat template)
#   RUN_ID   : name of the output directory under eval_outputs/ (refuses to reuse one that has a different MODEL)
#   DATASETS : space-separated benchmark names; default = all 8
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
: "${MODEL:?set MODEL}"; : "${RUN_ID:?set RUN_ID}"
# twin guard (same job name submitted to several partitions): first to start wins, pending twins are cancelled
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" != "rsft_bootstrap" ]; then
  OTHERS=$(squeue -u yuanzhou1 -h -n "$SLURM_JOB_NAME" -o "%i %T" | grep -v "^$SLURM_JOB_ID " || true)
  if echo "$OTHERS" | grep -q RUNNING; then echo "twin already running: $OTHERS; exiting"; exit 0; fi
  for j in $(echo "$OTHERS" | awk '$2=="PENDING"{print $1}'); do scancel $j && echo "cancelled pending twin $j"; done
fi
DATASETS=${DATASETS:-"aime amc cn_math_2024 gpqa kaoyan math minerva olympiadbench"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
# optional sharding of one benchmark by question index [START, END): separate result file, separate summary row
START=${START:-0}; END=${END:--1}
# optional generation seed (official protocol: 0). Use a distinct RUN_ID per seed: result filenames do not encode the seed.
SEED=${SEED:-0}
export HF_HOME=$P/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/eval_env

EVAL=$P/code/Project-Reasoning-SFT-LLM/eval
OUT=$P/eval_outputs/$RUN_ID
mkdir -p $OUT
cd $EVAL
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# manifest: refuse to mix two different models under one RUN_ID
if [ -f $OUT/manifest.json ]; then
  OLD=$(python -c "import json;print(json.load(open('$OUT/manifest.json'))['model'])")
  if [ "$OLD" != "$MODEL" ]; then echo "ERROR: $OUT already holds results for $OLD, not $MODEL"; exit 9; fi
else
  python - "$MODEL" "$OUT" <<'PY'
import json, sys, os, hashlib, subprocess, datetime
model, out = sys.argv[1], sys.argv[2]
P = "/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft"
def h(p):
    if not os.path.isfile(p): return None
    return hashlib.md5(open(p, "rb").read()).hexdigest()
cfg_hash = h(os.path.join(model, "config.json")) if os.path.isdir(model) else None
idx_hash = h(os.path.join(model, "model.safetensors.index.json")) if os.path.isdir(model) else None
man = {"model": model, "config_md5": cfg_hash, "safetensors_index_md5": idx_hash,
       "settings": {"temperature": 0.6, "top_p": 0.95, "n_sampling": 8, "max_tokens": 32768, "seed": int(os.environ.get("SEED", "0")),
                    "prompt_type": "qwen-instruct", "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}."},
       "eval_commit": subprocess.run(["git", "-C", f"{P}/code/Project-Reasoning-SFT-LLM", "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip(),
       "env_freeze": f"{P}/envs/eval_env_freeze.txt", "created": datetime.datetime.now().isoformat(),
       "jobs": []}
json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
PY
fi
python - "$OUT" "$SLURM_JOB_ID" "$DATASETS" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json")); m["jobs"].append({"job": sys.argv[2], "datasets": sys.argv[3]})
json.dump(m, open(f"{sys.argv[1]}/manifest.json", "w"), indent=1)
PY

declare -A SIZES=( [aime]=30 [amc]=40 [cn_math_2024]=30 [gpqa]=198 [kaoyan]=199 [math]=500 [minerva]=272 [olympiadbench]=675 )   # math: 500 records (README says 499 = line count without trailing newline)
FAIL=0
for DATA in $DATASETS; do
  KEY=$DATA; EXPECTED=${SIZES[$DATA]}; SHARDARG=""
  if [ "$END" != "-1" ] || [ "$START" != "0" ]; then KEY="${DATA}@s${START}-e${END}"; EXPECTED=$((END - START)); SHARDARG="--shard s${START}_e${END}"; fi
  echo "=============== $KEY  $(date)"
  # resume / cross-cluster idempotency: skip a benchmark already recorded complete in summary.tsv
  if [ -f $OUT/summary.tsv ] && awk -F'\t' -v d="$KEY" '$1==d && $NF=="True"{f=1} END{exit !f}' $OUT/summary.tsv; then
    echo "$KEY already complete in $OUT/summary.tsv; skipping"; continue
  fi
  T0=$(date +%s)
  set +e
  # run in its own process group so a hung exit (vLLM engine / grader pool not terminating) can be killed as a unit
  setsid python eval.py --model_name_or_path "$MODEL" --data_name "$DATA" --prompt_type "qwen-instruct" \
    --temperature 0.6 --top_p 0.95 --start_idx $START --end_idx $END --n_sampling 8 --k 1 --split test \
    --max_tokens 32768 --seed $SEED --surround_with_messages --output_dir "$OUT" \
    --completions_save_dir "$OUT/completions" --gpu_memory_utilization "$GPU_MEM_UTIL" > "$OUT/log_${KEY}.txt" 2>&1 &
  PID=$!
  RC=""; DONE_AT=""
  while kill -0 $PID 2>/dev/null; do
    sleep 20
    # watchdog: scores printed means the work is done; if the process is still alive 5 min later, it is hung at exit
    if [ -z "$DONE_AT" ] && grep -q "^Pass@1:" "$OUT/log_${KEY}.txt" 2>/dev/null; then DONE_AT=$(date +%s); fi
    if [ -n "$DONE_AT" ] && [ $(( $(date +%s) - DONE_AT )) -gt 300 ]; then
      echo "WATCHDOG: scores logged $(( $(date +%s) - DONE_AT ))s ago but process $PID still alive; killing process group" | tee -a "$OUT/log_${KEY}.txt"
      kill -TERM -- -$PID 2>/dev/null; sleep 15; kill -KILL -- -$PID 2>/dev/null
      RC=0; break
    fi
  done
  if [ -z "$RC" ]; then wait $PID; RC=$?; fi
  tail -n 4 "$OUT/log_${KEY}.txt"
  # a result file + scores in the log means success even if the exit was forced
  if [ $RC -ne 0 ] && grep -q "^Pass@1:" "$OUT/log_${KEY}.txt" && [ -n "$(find "$OUT" -path '*/completions' -prune -o -path "*/$DATA/test_*.jsonl" -print | head -1)" ]; then echo "exit $RC after scores were logged; treating as success"; RC=0; fi
  set -e
  T1=$(date +%s)
  echo "eval.py exit=$RC elapsed=$((T1-T0))s" | tee -a "$OUT/log_${KEY}.txt"
  if [ $RC -ne 0 ]; then FAIL=1; continue; fi
  python $P/code/scripts/eval_summary.py --run-dir "$OUT" --data "$DATA" --key "$KEY" $SHARDARG --expected "$EXPECTED" --elapsed $((T1-T0)) || FAIL=1
done
echo "=============== summary"; cat $OUT/summary.tsv 2>/dev/null || true
exit $FAIL
