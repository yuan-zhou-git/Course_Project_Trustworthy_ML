#!/bin/bash
#SBATCH --job-name=rsft_bootstrap
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# One queue wait instead of three: env setup -> data prep -> base-model eval pilot (amc), all on one A100.
set -euo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
# done marker: bootstrap may be queued on several clusters (Tinkercliffs, Falcon); once one finished, the rest exit 0
if [ -f $P/logs/bootstrap.done ]; then echo "bootstrap already done: $(cat $P/logs/bootstrap.done)"; exit 0; fi
if [ -f $P/logs/bootstrap.running ]; then echo "bootstrap running elsewhere: $(cat $P/logs/bootstrap.running); exiting"; exit 0; fi
echo "$SLURM_CLUSTER_NAME $SLURM_JOB_ID $(hostname) $(date)" > $P/logs/bootstrap.running
trap 'rm -f $P/logs/bootstrap.running' EXIT
# twin guard: this job is submitted to both the A100 and the H200 partition; whichever starts first wins
OTHERS=$(squeue -u yuanzhou1 -h -n "$SLURM_JOB_NAME" -o "%i %T" | grep -v "^$SLURM_JOB_ID " || true)
if echo "$OTHERS" | grep -q RUNNING; then echo "twin already running: $OTHERS; exiting"; exit 0; fi
for j in $(echo "$OTHERS" | awk '$2=="PENDING"{print $1}'); do scancel $j && echo "cancelled pending twin $j"; done
cd $P/slurm
echo "######## STAGE 1 setup_env  $(date)"
bash setup_env.slurm
echo "######## STAGE 2 prep_data  $(date)"
bash prep_data.slurm
echo "######## STAGE 3 eval pilot (base model, amc)  $(date)"
MODEL=Qwen/Qwen2.5-3B-Instruct RUN_ID=pilot_base DATASETS="amc" bash eval_run.slurm
echo "$SLURM_CLUSTER_NAME $SLURM_JOB_ID $(hostname) $(date)" > $P/logs/bootstrap.done
echo "######## BOOTSTRAP DONE $(date)"
