#!/bin/bash
#SBATCH --job-name=rsft_pipe_short
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --gres=gpu:4
#SBATCH --time=13:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err
#SBATCH --open-mode=append

# Thin submit wrapper: the pipeline body is read from disk when the job STARTS (Slurm snapshots only this file at submit
# time), so fixes to pipeline_tc.slurm made while the job waits in the queue still apply.
type module >/dev/null 2>&1 && export -f module
exec bash /projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/slurm/pipeline_tc.slurm
