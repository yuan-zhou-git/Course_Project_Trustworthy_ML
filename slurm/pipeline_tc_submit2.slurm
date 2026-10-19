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
#SBATCH --time=12:00:00
#SBATCH --exclude=tc-gpu004
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err
#SBATCH --open-mode=append

# Takeover copy of the short-trace pipeline. The first copy (8051533) landed on tc-gpu004, whose A100s are power-brake
# throttled (clocks_throttle_reasons 0x88, ~1155 MHz, ~110 W): training ran at ~40% speed and could not finish in its limit.
# 1. If THIS node's GPUs are power-brake throttled too, resubmit this wrapper excluding the node and leave the old copy running.
# 2. Otherwise stop the old copy, wait until it is gone, and run the pipeline body; train.slurm resumes from the old copy's
#    last complete checkpoint (same 4-GPU world size, same scratch).
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
OLD=${PIPE_OLD_JOB:-8051533}
EXC=${PIPE_EXCLUDE:-tc-gpu004}
echo "=== [$(date '+%F %T')] takeover job $SLURM_JOB_ID on $(hostname -s); excluded so far: $EXC"
THR=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader | tr '\n' ' ')
echo "GPU throttle reasons: $THR"
BRAKE=0; for t in $THR; do [ $(( t & 0x80 )) -ne 0 ] && BRAKE=1; done
if [ $BRAKE -eq 1 ]; then
  NEWEXC="$EXC,$(hostname -s)"
  j=$(PIPE_EXCLUDE=$NEWEXC PIPE_OLD_JOB=$OLD sbatch --parsable --exclude=$NEWEXC $P/slurm/pipeline_tc_submit2.slurm)
  echo "power-brake throttling on $(hostname -s): resubmitted as $j (exclude $NEWEXC); old copy $OLD left running"
  exit 0
fi
if squeue -h -j $OLD 2>/dev/null | grep -q .; then
  echo "stopping $OLD (throttled node) and continuing from its last complete checkpoint"
  scancel $OLD
  while squeue -h -j $OLD 2>/dev/null | grep -q .; do sleep 10; done
  sleep 30
fi
type module >/dev/null 2>&1 && export -f module
exec bash $P/slurm/pipeline_tc.slurm
