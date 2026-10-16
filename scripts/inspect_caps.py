#!/usr/bin/env python
"""Why do fine-tuned answers hit the 32K cap? Classify every generated sample of each run/benchmark.

Reads the merged result records (question, generated_responses, answers_correctness) and, when available, the vLLM
completion pickles (finish_reason, token counts). Per sample it records:
  capped        finish_reason == length (or >= max tokens)
  has_boxed     a \\boxed{ appears in the response
  think_closed  a </think> tag appears (the trace reached its conclusion phase)
  repetitive    the last 3000 chars contain a 120-char substring that occurs >= 3 times (degenerate loop)
and aggregates per (run, benchmark): fractions, plus accuracy of capped vs uncapped samples.
Stdlib only, but the pickles need the eval env (vLLM classes) -> run inside the finalize job.
"""
import argparse, glob, json, os, pickle, re, sys, collections

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="label=run_id")
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=32768)
a = ap.parse_args()
runs = dict(r.split("=", 1) for r in a.runs)
BENCH = ["aime", "math", "cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]

def repetitive(text, tail=3000, k=120, reps=3):
    t = text[-tail:]
    if len(t) < k * reps:
        return False
    seen = collections.Counter(t[i:i + k] for i in range(0, len(t) - k, 40))
    return any(c >= reps for c in seen.values())

def shard_key(f):
    m = re.search(r"_s(\d+)_e(-?\d+)\.jsonl$", os.path.basename(f))
    return (int(m.group(1)), int(m.group(2))) if m else (0, -1)

summary = {}
md = ["# Stopping-failure inspection\n", "| run | benchmark | samples | capped | capped & no `\\boxed{}` | capped & `</think>` never reached | capped & repetitive tail | acc uncapped | acc capped |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for lab, rid in runs.items():
    for b in BENCH:
        files = [f for f in glob.glob(f"{a.root}/{rid}/**/{b}/test_*.jsonl", recursive=True) if "/completions/" not in f.replace("\\", "/")]
        if not files:
            continue
        # narrowest shard per index (same rule as aggregate_results.py)
        shards = sorted(((e if e != -1 else 10**9) - s, s, f) for f in files for (s, e) in [shard_key(f)])
        owner = {}
        for w, s, f in shards:
            for idx in range(s, s + min(w, 10**6)):
                owner.setdefault(idx, f)
        # finish reasons / lengths from pickles, keyed by (shard start, question pos, sample)
        fin = {}
        for pk in glob.glob(f"{a.root}/{rid}/completions/**/{b}/*.pkl", recursive=True):
            m = re.search(r"_s(\d+)_e(-?\d+)_gen_round(\d+)\.pkl$", os.path.basename(pk))
            if not m: continue
            s0 = int(m.group(1))
            try:
                comps = pickle.load(open(pk, "rb"))
            except Exception as ex:  # noqa
                print("could not read", pk, ex, file=sys.stderr); continue
            for qi, c in enumerate(comps):
                for si, o in enumerate(c.outputs):
                    fin[(s0, qi, si)] = (getattr(o, "finish_reason", None), len(o.token_ids))
        n = cap = cap_nobox = cap_nothink = cap_rep = 0
        corr_unc = tot_unc = corr_cap = tot_cap = 0
        for w, s, f in shards:
            for qi, l in enumerate(open(f, encoding="utf-8")):
                if owner.get(s + qi) != f:
                    continue
                r = json.loads(l)
                for si, (resp, ok) in enumerate(zip(r["generated_responses"], r["answers_correctness"])):
                    fr = fin.get((s, qi, si))
                    is_cap = (fr[0] == "length" or fr[1] >= a.max_tokens) if fr else (len(resp) > 60000 and "\\boxed" not in resp[-2000:])
                    n += 1
                    if is_cap:
                        cap += 1; tot_cap += 1; corr_cap += bool(ok)
                        cap_nobox += "\\boxed" not in resp
                        cap_nothink += "</think>" not in resp
                        cap_rep += repetitive(resp)
                    else:
                        tot_unc += 1; corr_unc += bool(ok)
        row = {"samples": n, "capped": cap, "capped_no_boxed": cap_nobox, "capped_think_open": cap_nothink, "capped_repetitive": cap_rep,
               "acc_uncapped": corr_unc / tot_unc if tot_unc else None, "acc_capped": corr_cap / tot_cap if tot_cap else None,
               "finish_reason_available": bool(fin)}
        summary.setdefault(lab, {})[b] = row
        pct = lambda x: f"{100 * x / cap:.0f}%" if cap else "-"
        md.append(f"| {lab} | {b} | {n} | {cap} ({100 * cap / n:.1f}%) | {pct(cap_nobox)} | {pct(cap_nothink)} | {pct(cap_rep)} | "
                  f"{(corr_unc / tot_unc) if tot_unc else float('nan'):.3f} | {(corr_cap / tot_cap) if tot_cap else float('nan'):.3f} |")
        print(md[-1], flush=True)
md.append("\nPercentages in the middle columns are fractions of the capped samples. `</think>` never reached = the model was still in its "
          "thinking phase at the cap; repetitive = a 120-char substring repeats >= 3 times in the last 3000 chars. A capped sample "
          "without `\\boxed{}` is graded wrong by construction.\n")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "stopping_failures.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump(summary, open(os.path.join(a.out, "stopping_failures.json"), "w"), indent=1)
print("written", os.path.join(a.out, "stopping_failures.md"))
