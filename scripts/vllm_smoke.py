#!/usr/bin/env python
"""Smallest end-to-end check of the eval env on a GPU node: torch sees the GPU, vLLM loads a model and samples.
Prints SMOKE_OK on success (callers grep for it, because vLLM can hang at interpreter exit).
Usage: python vllm_smoke.py <model id or dir> [gpu_memory_utilization]
The __main__ guard matters: with VLLM_WORKER_MULTIPROC_METHOD=spawn (as in eval_run.slurm) the engine process
re-imports this module.
"""
import sys


def main():
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), flush=True)
    assert torch.cuda.is_available(), "torch cannot see a GPU (driver too old for this CUDA build?)"
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), flush=True)
    import vllm
    from vllm import LLM, SamplingParams
    print("vllm", vllm.__version__, flush=True)
    llm = LLM(model=sys.argv[1], gpu_memory_utilization=float(sys.argv[2]) if len(sys.argv) > 2 else 0.3, max_model_len=4096, seed=0)
    out = llm.generate(["What is 17*23? Answer with the number only."],
                       SamplingParams(temperature=0.6, top_p=0.95, max_tokens=64, n=2, seed=0))
    print("SMOKE_OK", [o.text.strip()[:60] for o in out[0].outputs], flush=True)


if __name__ == "__main__":
    main()
