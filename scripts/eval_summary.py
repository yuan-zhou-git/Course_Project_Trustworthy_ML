#!/usr/bin/env python
"""Completeness check + generation-length summary for one benchmark of one eval run.

Appends a row to <run-dir>/summary.tsv:
  data  n_questions  n_responses_per_q  acc(pass@8)  pass@1  mean_gen_tokens  p95_gen_tokens  frac_len_capped  elapsed_s
Exits non-zero if the question count or the responses-per-question do not match the expectation.
"""
import argparse, glob, json, os, pickle, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--expected", type=int, required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--elapsed", type=int, default=-1)
a = ap.parse_args()

files = [f for f in glob.glob(f"{a.run_dir}/*/*/{a.data}/test_*.jsonl") + glob.glob(f"{a.run_dir}/*/{a.data}/test_*.jsonl")
         if "completions" not in f]
if not files:
    print(f"[summary] ERROR no result jsonl for {a.data} under {a.run_dir}"); sys.exit(4)
res = files[0]
rows = [json.loads(l) for l in open(res, encoding="utf-8")]
nq = len(rows)
per_q = sorted({len(r.get("generated_responses", [])) for r in rows})
ok = nq == a.expected and per_q == [a.n]
print(f"[summary] {a.data}: {nq} questions (expected {a.expected}), responses/question {per_q} (expected [{a.n}])")

# metrics recomputed from the per-sample correctness (the log has no scores when eval.py skipped a cached result)
acc = p1 = float("nan")
if rows and all("answers_correctness" in r for r in rows):
    acc = sum(1 for r in rows if any(r["answers_correctness"])) / nq
    p1 = sum(sum(bool(x) for x in r["answers_correctness"]) / len(r["answers_correctness"]) for r in rows) / nq
log = f"{a.run_dir}/log_{a.data}.txt"
if os.path.isfile(log):
    txt = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"^Acc:\s*([0-9.]+)", txt, re.M)
    if m and abs(float(m[-1]) - acc) > 1e-3:
        print(f"[summary] WARNING log Acc {m[-1]} differs from recomputed {acc:.4f}")

# generation lengths and finish reasons from the saved vLLM completions (pickles, one per generation round)
lens, capped, total = [], 0, 0
for pk in sorted(glob.glob(f"{a.run_dir}/completions/**/{a.data}/*.pkl", recursive=True)):
    try:
        comps = pickle.load(open(pk, "rb"))
    except Exception as e:  # noqa
        print(f"[summary] could not read {pk}: {e}"); continue
    for c in comps:
        for o in c.outputs:
            n = len(o.token_ids); lens.append(n); total += 1
            if getattr(o, "finish_reason", None) == "length" or n >= a.max_tokens:
                capped += 1
mean = sum(lens) / len(lens) if lens else float("nan")
p95 = sorted(lens)[int(0.95 * (len(lens) - 1))] if lens else float("nan")
frac = capped / total if total else float("nan")
print(f"[summary] gen tokens mean={mean:.0f} p95={p95} capped={capped}/{total} ({frac:.3f})")

hdr = "data\tn_questions\tresp_per_q\tacc_pass8\tpass1\tmean_gen_tokens\tp95_gen_tokens\tfrac_len_capped\telapsed_s\tcomplete\n"
path = f"{a.run_dir}/summary.tsv"
new = not os.path.isfile(path)
with open(path, "a") as f:
    if new: f.write(hdr)
    f.write(f"{a.data}\t{nq}\t{','.join(map(str, per_q))}\t{acc:.4f}\t{p1:.4f}\t{mean:.0f}\t{p95}\t{frac:.3f}\t{a.elapsed}\t{ok}\n")
sys.exit(0 if ok else 5)
