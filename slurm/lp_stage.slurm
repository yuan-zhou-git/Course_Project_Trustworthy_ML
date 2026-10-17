#!/bin/bash
#SBATCH --job-name=rsft_lp
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# One generic 1-GPU stage of the learning-progress pipeline. STAGE = build | score | select
#   build : probe2k.json + lp_candidates.json + lp_meta.json (needs the pool and the tokenizer)
#   score : MODEL=<hf id or dir> TAG=<base|probe> SHARD=k/n  -> data/lp/scores_$TAG/shard_k.jsonl
#   select: lp15k.json + manifest from the two score sets
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
: "${STAGE:?set STAGE}"
export HF_HOME=$P/hf_cache
export HF_HUB_OFFLINE=1
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/train_env
LP=$P/data/lp; mkdir -p $LP
case "$STAGE" in
  build)
    python $P/code/scripts/make_probe_and_candidates.py --pool $P/data/raw/data/acereason11_100k.json \
      --dev-panel $P/data/selections/dev200.jsonl --out-dir $LP --n-probe 2000
    cp $LP/probe2k.json $P/data/selections/probe2k.json
    cat $LP/lp_build_manifest.json | head -20 ;;
  score)
    : "${MODEL:?}"; : "${TAG:?}"; : "${SHARD:?}"
    mkdir -p $LP/scores_$TAG
    python $P/code/scripts/score_loss.py --model "$MODEL" --pool $P/data/raw/data/acereason11_100k.json \
      --cands $LP/lp_candidates.json --shard $SHARD --out $LP/scores_$TAG/shard_${SHARD%%/*}.jsonl ;;
  select)
    python $P/code/scripts/select_lp.py --pool $P/data/raw/data/acereason11_100k.json --lp-dir $LP --out $P/data/selections \
      --manifests $P/data/selections/random15k_manifest.json $P/data/selections/advanced15k_manifest.json
    md5sum $P/data/selections/lp15k.json ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "=== LP STAGE $STAGE DONE $(date)"
