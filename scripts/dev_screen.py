#!/usr/bin/env python
"""Screen a checkpoint on the held-out development panel with the official decoding settings but n samples per question.

Same prompt construction as the course harness (system prompt + question, Qwen chat template), temperature 0.6, top-p 0.95,
max 32,768 tokens, seed 0; grading = extract_answer(..., "math") + check_is_correct from the course utils (teacher's boxed
answer as reference). Appends one row per checkpoint to <out>/dev_screen.tsv and writes per-question details.
Must run from the course eval/ directory (imports utils.*).
"""
import argparse, json, os, sys, time, statistics
sys.path.insert(0, os.getcwd())
from utils.parser import extract_answer, strip_string  # noqa: E402
from utils.grader import check_is_correct  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--panel", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
panel = [json.loads(l) for l in open(a.panel, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
                                   tokenize=False, add_generation_prompt=True) for q in panel]
llm = LLM(model=a.model, trust_remote_code=True, gpu_memory_utilization=a.gpu_mem, seed=a.seed, tensor_parallel_size=1)
sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.n, seed=a.seed)
t0 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t0
rows, lens, capped, corr, anyc = [], [], 0, 0, 0
for q, o in zip(panel, outs):
    flags, preds = [], []
    for s in o.outputs:
        lens.append(len(s.token_ids)); capped += (s.finish_reason == "length")
        pred = extract_answer(s.text, "math")
        ok = bool(check_is_correct(pred, q["answer"])) if pred else False
        flags.append(ok); preds.append(pred)
    corr += sum(flags); anyc += any(flags)
    rows.append({"pos": q["pos"], "answer": q["answer"], "preds": preds, "correct": flags,
                 "finish": [s.finish_reason for s in o.outputs], "tokens": [len(s.token_ids) for s in o.outputs]})
N = len(panel) * a.n
res = {"label": a.label, "model": a.model, "questions": len(panel), "n": a.n, "acc_per_sample": corr / N, "pass_at_n": anyc / len(panel),
       "frac_capped": capped / N, "mean_tokens": statistics.mean(lens), "p95_tokens": sorted(lens)[int(0.95 * (len(lens) - 1))],
       "gen_seconds": gen_s, "seed": a.seed}
json.dump({"summary": res, "rows": rows}, open(os.path.join(a.out, f"dev_{a.label}.json"), "w"), indent=1)
path = os.path.join(a.out, "dev_screen.tsv")
new = not os.path.isfile(path)
with open(path, "a") as f:
    if new: f.write("label\tquestions\tn\tacc_per_sample\tpass_at_n\tfrac_capped\tmean_tokens\tp95_tokens\tgen_seconds\n")
    f.write(f"{a.label}\t{len(panel)}\t{a.n}\t{res['acc_per_sample']:.4f}\t{res['pass_at_n']:.4f}\t{res['frac_capped']:.4f}\t{res['mean_tokens']:.0f}\t{res['p95_tokens']}\t{gen_s:.0f}\n")
print(json.dumps(res, indent=1))
