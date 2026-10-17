#!/bin/bash
#SBATCH --job-name=rsft_screen
#SBATCH --account=tml_2026
#SBATCH --partition=l40s_normal_q
#SBATCH --qos=fal_l40s_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Checkpoint screening on the held-out dev panel (no benchmark data). Builds the panel once, then screens each
# label=path pair in MODELS with dev_screen.py (official decoding, n=2 samples). A per-model watchdog kills a run that
# hangs after writing its JSON (same failure mode as the course harness).
# Usage: MODELS="adv_ep1=$P/models/advanced15k/checkpoint-1875 adv_ep2=..." sbatch slurm/screen.slurm
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
: "${MODELS:?set MODELS as space-separated label=path}"
export HF_HOME=$P/hf_cache
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/eval_env
PANEL=$P/data/dev_panel/dev200.jsonl
OUT=$P/eval_outputs/dev_screen
mkdir -p $(dirname $PANEL) $OUT
if [ ! -s $PANEL ]; then
  python $P/code/scripts/make_dev_panel.py --pool $P/data/raw/data/acereason11_100k.json \
    --manifests $P/data/selections/random15k_manifest.json $P/data/selections/advanced15k_manifest.json --out $PANEL --n 200
  cp $PANEL $P/data/selections/dev200.jsonl
fi
cd $P/code/Project-Reasoning-SFT-LLM/eval
for spec in $MODELS; do
  LABEL=${spec%%=*}; MODEL=${spec#*=}
  if [ -s $OUT/dev_$LABEL.json ]; then echo "$LABEL already screened; skipping"; continue; fi
  echo "=============== $LABEL  $MODEL  $(date)"
  set +e
  setsid python $P/code/scripts/dev_screen.py --model "$MODEL" --label "$LABEL" --panel $PANEL --out $OUT --n 2 > $OUT/log_$LABEL.txt 2>&1 &
  PID=$!; DONE_AT=""
  while kill -0 $PID 2>/dev/null; do
    sleep 20
    [ -z "$DONE_AT" ] && [ -s $OUT/dev_$LABEL.json ] && DONE_AT=$(date +%s)
    if [ -n "$DONE_AT" ] && [ $(( $(date +%s) - DONE_AT )) -gt 180 ]; then echo "WATCHDOG: killing $LABEL after its result was written"; kill -TERM -- -$PID 2>/dev/null; sleep 10; kill -KILL -- -$PID 2>/dev/null; fi
  done
  set -e
  tail -n 3 $OUT/log_$LABEL.txt
done
echo "=== dev_screen.tsv"; cat $OUT/dev_screen.tsv
echo "=== SCREEN DONE $(date)"
