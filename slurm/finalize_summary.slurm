#!/bin/bash
#SBATCH --job-name=rsft_finalize
#SBATCH --account=tml_2026
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Writes any missing summary.tsv rows for every result file found under the given run dirs (full runs and shards),
# then runs the aggregation. Needs the eval env only to unpickle vLLM completion objects (lengths / finish reasons).
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
RUNS=${RUNS:-"random15k_v1 advanced15k_v1"}
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/eval_env
declare -A SIZES=( [aime]=30 [amc]=40 [cn_math_2024]=30 [gpqa]=198 [kaoyan]=199 [math]=500 [minerva]=272 [olympiadbench]=675 )
for R in $RUNS; do
  OUT=$P/eval_outputs/$R
  find "$OUT" -path '*/completions' -prune -o -name 'test_*.jsonl' -print | sort | while read -r f; do
    DATA=$(basename "$(dirname "$f")")
    base=$(basename "$f" .jsonl)                       # test_qwen-instruct_t0.6_k8_s<START>_e<END>
    S=$(echo "$base" | sed -E 's/.*_s([0-9]+)_e(-?[0-9]+)$/\1/'); E=$(echo "$base" | sed -E 's/.*_s([0-9]+)_e(-?[0-9]+)$/\2/')
    if [ "$S" = "0" ] && { [ "$E" = "-1" ] || [ "$E" = "${SIZES[$DATA]}" ]; }; then KEY=$DATA; EXP=${SIZES[$DATA]}; SHARD=""; else KEY="${DATA}@s${S}-e${E}"; EXP=$((E - S)); SHARD="--shard s${S}_e${E}"; fi
    if [ -f $OUT/summary.tsv ] && awk -F'\t' -v d="$KEY" '$1==d && $NF=="True"{x=1} END{exit !x}' $OUT/summary.tsv; then echo "$R $KEY: row present"; continue; fi
    echo "$R $KEY: writing row"
    python $P/code/scripts/eval_summary.py --run-dir "$OUT" --data "$DATA" --key "$KEY" $SHARD --expected "$EXP" --elapsed -1 || echo "  (incomplete)"
  done
  echo "== $R summary.tsv"; cat $OUT/summary.tsv 2>/dev/null || echo "(no rows yet)"
done
python $P/code/scripts/aggregate_results.py --root $P/eval_outputs --runs base=base_v1 random=random15k_v1 advanced=advanced15k_v1 --out $P/reports --B 2000 | tail -60
echo "=== FINALIZE DONE $(date)"
