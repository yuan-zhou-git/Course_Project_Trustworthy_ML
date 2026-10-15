#!/bin/bash
#SBATCH --job-name=rsft_train
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# LLaMA-Factory full SFT. One Slurm task; torchrun (via FORCE_TORCHRUN) spawns one worker per allocated GPU.
# Usage: RUN=random15k CONFIG=yamls/random15k.yaml sbatch --job-name=rsft_train_random --time=... train.slurm
#   RUN     : run name; output_dir in the yaml must be /scratch/yuanzhou1/reasoning_sft/ckpt/$RUN
#   CONFIG  : yaml path relative to $P/code
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
S=/scratch/yuanzhou1/reasoning_sft
: "${RUN:?set RUN}"; : "${CONFIG:?set CONFIG}"
# twin guard (same job name submitted to several partitions): first to start wins, pending twins are cancelled
OTHERS=$(squeue -u yuanzhou1 -h -n "$SLURM_JOB_NAME" -o "%i %T" | grep -v "^$SLURM_JOB_ID " || true)
if echo "$OTHERS" | grep -q RUNNING; then echo "twin already running: $OTHERS; exiting"; exit 0; fi
for j in $(echo "$OTHERS" | awk '$2=="PENDING"{print $1}'); do scancel $j && echo "cancelled pending twin $j"; done
export HF_HOME=$P/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$P/logs/wandb
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
mkdir -p $WANDB_DIR
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/train_env
# Liger kernel (fused linear cross-entropy): avoids materializing the 16K x 152K-vocab logits that OOM a 44 GB L40S
python -c "import liger_kernel" 2>/dev/null || { pip install -q liger-kernel && pip freeze > $P/envs/train_env_freeze.txt; }
python -c "import liger_kernel, importlib.metadata as m; print('liger-kernel', m.version('liger-kernel'))"
# wandb (offline mode; synced later once the user has logged in)
python -c "import wandb" 2>/dev/null || { pip install -q wandb && pip freeze > $P/envs/train_env_freeze.txt; }
python -c "import wandb; print('wandb', wandb.__version__)"

echo "=== job $SLURM_JOB_ID run=$RUN config=$CONFIG"; date; hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  cpus=$SLURM_CPUS_PER_TASK"

# fast IO: training data on scratch, checkpoints on scratch
mkdir -p $S/data $S/ckpt/$RUN; touch $S/ckpt/$RUN/.keep
rsync -a $P/data/selections/ $S/data/
cp $P/envs/train_env_freeze.txt $S/ckpt/$RUN/
(cd $P/code/LLaMA-Factory && git log -1 --format="LLaMA-Factory %H") > $S/ckpt/$RUN/commits.txt

# global batch fixed at 8 = NGPU x micro-batch 1 x grad-accum; the yaml's grad-accum is rewritten accordingly
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GA=$(( 8 / NGPU )); [ $GA -lt 1 ] && GA=1
sed -E "s/^gradient_accumulation_steps: .*/gradient_accumulation_steps: $GA/" $P/code/$CONFIG > $S/ckpt/$RUN/config_used.yaml
echo "NGPU=$NGPU grad_accum=$GA"; grep -E "gradient_accumulation|per_device_train|deepspeed|cutoff_len|num_train_epochs|max_steps" $S/ckpt/$RUN/config_used.yaml
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=$NGPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
cd $P/code/LLaMA-Factory
START=$(date +%s)
set +e
FORCE_TORCHRUN=1 llamafactory-cli train $S/ckpt/$RUN/config_used.yaml
RC=$?
set -e
END=$(date +%s)
echo "=== train exit code $RC after $((END-START)) s"

# stage results to the project folder (skip DeepSpeed optimizer shards)
mkdir -p $P/models/$RUN
rsync -a --exclude 'global_step*' --exclude '*.pt' $S/ckpt/$RUN/ $P/models/$RUN/
echo "=== staged to $P/models/$RUN"; du -sh $P/models/$RUN; ls $P/models/$RUN
exit $RC
