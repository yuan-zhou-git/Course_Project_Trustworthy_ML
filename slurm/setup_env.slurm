#!/bin/bash
#SBATCH --job-name=rsft_setup
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.out
#SBATCH --error=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft/logs/%x.%j.err

# Builds TWO frozen conda environments for the Reasoning-SFT course project:
#   train_env : LLaMA-Factory + FlashAttention-2 + DeepSpeed 0.16.8   (transformers pinned by LLaMA-Factory)
#   eval_env  : vLLM + rule-based grader deps                          (vLLM pins its own torch/transformers)
# Two envs because LLaMA-Factory pins transformers<=4.55 while current vLLM needs a newer one.
# Runs on a GPU node (nothing heavy on the login node). Idempotent: re-run to finish a partial install.
set -euo pipefail
set -x

P=/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft
S=/scratch/yuanzhou1/reasoning_sft
export HF_HOME=$P/hf_cache
export PIP_CACHE_DIR=$S/pip_cache
export CONDA_PKGS_DIRS=$P/envs/conda_pkgs
export TMPDIR=$S/tmp
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR" "$HF_HOME"

echo "=== node info"; hostname; nvidia-smi; date

module reset
module load Miniconda3/25.11.1-1
module load CUDA/12.8.0
nvcc --version | tail -1

############ train_env ############
TENV=$P/envs/train_env
# conda-forge only: the default anaconda.com channels require an interactive Terms-of-Service acceptance on Miniconda 25.x
if [ ! -x "$TENV/bin/python" ]; then
  conda create -y -p "$TENV" -c conda-forge --override-channels python=3.10
fi
source activate "$TENV"
python -V
pip install --upgrade pip wheel setuptools
cd $P/code/LLaMA-Factory
pip install -r requirements.txt
pip install -e ".[torch,metrics]" --no-build-isolation
# Pin torch to a CUDA-12.8 build: the newest torch is built for CUDA 13 (no CUDA 13 module here, no flash-attn wheel).
pip install --force-reinstall --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2.0.0" sympy filelock typing_extensions networkx jinja2 fsspec   # torch runtime deps (no-deps above)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda.startswith('12.')"
# FlashAttention-2 prebuilt wheel matching torch 2.8 / cu12 / cp310 / cxx11abi
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install "$FA_WHL" || { export MAX_JOBS=8; pip install flash-attn==2.8.3 --no-build-isolation; }
pip install deepspeed==0.16.8
python - <<'PY'
import torch, importlib
print("TRAIN torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ["flash_attn", "deepspeed", "llamafactory", "transformers", "accelerate", "trl", "datasets"]:
    mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
import flash_attn_2_cuda  # noqa
print("flash_attn kernel import ok")
PY
pip freeze > $P/envs/train_env_freeze.txt
conda deactivate

############ eval_env ############
EENV=$P/envs/eval_env
if [ ! -x "$EENV/bin/python" ]; then
  conda create -y -p "$EENV" -c conda-forge --override-channels python=3.10
fi
source activate "$EENV"
python -V
pip install --upgrade pip wheel setuptools
pip install vllm word2number
pip install --no-deps latex2sympy2==1.9.1
# latex2sympy2 1.9.1 ships ANTLR ATN version 3: runtime 4.9.3 reads it (4.11 does not). sympy stays at torch's >=1.13.
pip install "antlr4-python3-runtime==4.9.3" Pebble timeout-decorator python_dateutil tqdm datasets huggingface_hub
pip install --upgrade "sympy>=1.13.3"
python - <<'PY'
import torch, importlib
print("EVAL torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ["vllm", "transformers", "sympy", "latex2sympy2", "word2number", "pebble", "timeout_decorator"]:
    mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
PY
pip freeze > $P/envs/eval_env_freeze.txt

############ prefetch public model + training pool ############
python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
P = "/projects/ai_wireless_hu_lab/yuanzhou/reasoning_sft"
p = snapshot_download("Qwen/Qwen2.5-3B-Instruct")
print("model at", p)
f = hf_hub_download(repo_id="redsgnaoh/acereason11_100k", repo_type="dataset",
                    filename="data/acereason11_100k.json", local_dir=f"{P}/data/raw")
print("dataset at", f, os.path.getsize(f))
PY
ls -la $P/data/raw/data/
echo "=== SETUP DONE"; date
