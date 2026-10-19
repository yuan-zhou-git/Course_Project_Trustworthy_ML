#!/bin/bash
#SBATCH --job-name=rsft_pipe_short2
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err
#SBATCH --open-mode=append

# Training-only continuation of the short-trace run on Tinkercliffs, queued behind the first copy (8051533, which reaches its
# time limit before the end of training). Any node, including the throttled tc-gpu004: the remaining ~1,250 steps fit even at
# throttled speed. Same job name as the takeover copy, so the pipeline's twin guard lets only one of them run. Submit with
# PIPE_STOP_AFTER_TRAIN=1 and --dependency=afterany:8051533; evaluation runs as Falcon shard jobs.
type module >/dev/null 2>&1 && export -f module
exec bash /projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/slurm/pipeline_tc.slurm
