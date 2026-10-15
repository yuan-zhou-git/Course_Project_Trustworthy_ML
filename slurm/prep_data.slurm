#!/bin/bash
#SBATCH --job-name=rsft_prep
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Builds random15k / advanced15k / pilot400 + manifests, and registers them for LLaMA-Factory.
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
export HF_HOME=$P/hf_cache
export TOKENIZERS_PARALLELISM=true
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/train_env
OUT=$P/data/selections
mkdir -p $OUT

python $P/code/scripts/prep_data.py \
  --pool $P/data/raw/data/acereason11_100k.json \
  --eval-data $P/code/Project-Reasoning-SFT-LLM/eval/data \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --out $OUT --n 15000 --seed 20261015 --cutoff 16384 ${PREP_EXTRA:-}

# pilot400: 200 longest (by chars) + 200 random rows of random15k; only for throughput/memory pilots
python - <<'PY'
import json, random
P="/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/data/selections"
d=json.load(open(f"{P}/random15k.json",encoding="utf-8"))
order=sorted(range(len(d)), key=lambda i: -len(d[i]["output"]))
longest=order[:200]; rest=order[200:]
rng=random.Random(7); pick=sorted(longest+rng.sample(rest,200))
json.dump([d[i] for i in pick], open(f"{P}/pilot400.json","w",encoding="utf-8"), ensure_ascii=False)
print("pilot400 written; longest output chars:", len(d[order[0]]["output"]))
PY
cp $P/code/data/dataset_info.json $OUT/dataset_info.json
ls -la $OUT
md5sum $OUT/*.json
echo "=== PREP DONE"
