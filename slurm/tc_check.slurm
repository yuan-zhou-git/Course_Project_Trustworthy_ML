#!/bin/bash
#SBATCH --job-name=rsft_tccheck
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Environment check on a Tinkercliffs A100 node: driver version, the CUDA 13.0 eval env (torch 2.13 + vLLM 0.31,
# never run on TC before), and the CUDA 12.8 train env. Short and small so it can backfill.
set -uo pipefail
P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
export HF_HOME=$P/hf_cache HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
date; hostname
nvidia-smi
type module >/dev/null 2>&1 && export -f module; bash -c "type module" >/dev/null 2>&1 && echo "module visible to child shells" || echo "WARNING: module NOT visible to child shells"
module reset; module load Miniconda3/25.11.1-1 CUDA/12.8.0
source activate $P/envs/eval_env
echo "--- diagnostics"; which gcc cc nvcc 2>&1; gcc --version 2>&1 | head -1; echo "TMPDIR=${TMPDIR:-unset}"; df -h /tmp | tail -1
ls -d $HOME/.cache/vllm $HOME/.cache/flashinfer $HOME/.triton 2>&1; free -g | head -2; ulimit -n -u 2>&1 | tr "
" " "; echo
SM=$P/logs/tccheck_smoke_$SLURM_JOB_ID.txt
timeout 900 python $P/code/scripts/vllm_smoke.py $P/models/advanced15k 0.5 > $SM 2>&1
echo "smoke exit $? (full log $SM)"; grep -E "^(torch|device|vllm|SMOKE_OK)" $SM
grep -nE "Error|error|Exception|failed|Failed" $SM | grep -v "^\s*$" | head -20 | cut -c1-300
conda deactivate
source activate $P/envs/train_env
python -c "import torch; assert torch.cuda.is_available(); print('train_env ok', torch.__version__, torch.cuda.get_device_name(0))"
echo "=== TC CHECK DONE $(date)"
