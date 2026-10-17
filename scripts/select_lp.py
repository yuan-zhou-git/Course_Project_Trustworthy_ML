#!/usr/bin/env python
"""Learning-progress selection, step 3: pick 15K examples from the scored candidates.

score_i = loss_before_i - loss_after_i  (mean response-token loss under base Qwen minus under the probe-trained Qwen).
1. GRAPE-style: among rows sharing a prompt, keep the response with the highest score (tie: lowest loss_after).
2. Coverage: strata = source x total-token band (<=4K, 4-8K, 8-16K); quotas proportional to the number of distinct
   prompts per stratum (same construction as the first selected set); within a stratum take the top-quota rows by score.
3. Write lp15k.json (alpaca) + manifest with score/loss quantiles of the selected set vs the candidate pool.
Pre-registered: the rule is fixed before any benchmark result of the resulting model is seen.
"""
import argparse, collections, glob, json, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--lp-dir", required=True, help="dir with lp_candidates.json, lp_meta.json, scores_base/, scores_probe/")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=15000)
ap.add_argument("--bands", default="4096,8192")
ap.add_argument("--cutoff", type=int, default=16384)
ap.add_argument("--manifests", nargs="*", default=[], help="other 15K manifests, to report overlap")
a = ap.parse_args()

meta = {int(k): v for k, v in json.load(open(f"{a.lp_dir}/lp_meta.json")).items()}
def read_scores(d):
    out = {}
    for f in glob.glob(f"{d}/*.jsonl"):
        for l in open(f):
            r = json.loads(l); out[r["pos"]] = r["mean_loss"]
    return out
lb, la = read_scores(f"{a.lp_dir}/scores_base"), read_scores(f"{a.lp_dir}/scores_probe")
common = [p for p in meta if p in lb and p in la]
print(f"candidates {len(meta)}, scored by both models {len(common)}")
score = {p: lb[p] - la[p] for p in common}

# 1. one response per prompt
by_prompt = collections.defaultdict(list)
for p in common:
    by_prompt[meta[p]["prompt_hash"]].append(p)
chosen_resp = {}
for h, ps in by_prompt.items():
    chosen_resp[h] = max(ps, key=lambda p: (score[p], -la[p]))
reps = sorted(chosen_resp.values())
print("distinct prompts", len(reps), "rows dropped as alternative responses", len(common) - len(reps))

# 2. strata and quotas
edges = [int(x) for x in a.bands.split(",")] + [a.cutoff]
def band(t):
    for k, e in enumerate(edges):
        if t <= e: return k
    return len(edges) - 1
strata = collections.defaultdict(list)
for p in reps:
    strata[(meta[p]["source"], band(meta[p]["total_tokens"]))].append(p)
keys = sorted(strata); E = len(reps)
raw = {k: a.n * len(strata[k]) / E for k in keys}
quota = {k: int(raw[k]) for k in keys}
for k in sorted(keys, key=lambda k: (-(raw[k] - quota[k]), k))[:a.n - sum(quota.values())]:
    quota[k] += 1
sel = []
for k in keys:
    ranked = sorted(strata[k], key=lambda p: (-score[p], la[p]))
    sel += ranked[:quota[k]]
sel = sorted(sel)
assert len(sel) == a.n, (len(sel), a.n)

pool = json.load(open(a.pool, encoding="utf-8"))
def alpaca(r):
    return {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"],
            "system": r.get("system", "Please reason step by step, and put your final answer within \\boxed{}.")}
json.dump([alpaca(pool[p]) for p in sel], open(f"{a.out}/lp15k.json", "w", encoding="utf-8"), ensure_ascii=False)

def q(xs):
    xs = sorted(xs); n = len(xs)
    return {str(t): round(xs[min(n - 1, int(t * n))], 4) for t in (0.05, 0.25, 0.5, 0.75, 0.95)} if n else {}
overlaps = {}
for m in a.manifests:
    rows = {r["pos"] for r in json.load(open(m))["rows"]}
    overlaps[m.split("/")[-1]] = len(rows & set(sel))
man = {"method": "learning progress: loss_base - loss_probe over response tokens; one response per prompt (max score); "
                 "stratified by source x token band with quotas proportional to distinct prompts; top-score within stratum",
       "n": len(sel), "rows": [{"pos": p, "index": None} for p in sel],
       "strata": [{"source": k[0], "band": k[1], "size": len(strata[k]), "quota": quota[k]} for k in keys],
       "score_quantiles_selected": q([score[p] for p in sel]), "score_quantiles_candidates": q([score[p] for p in reps]),
       "loss_base_selected_mean": statistics.mean(lb[p] for p in sel), "loss_probe_selected_mean": statistics.mean(la[p] for p in sel),
       "loss_base_candidates_mean": statistics.mean(lb[p] for p in reps), "loss_probe_candidates_mean": statistics.mean(la[p] for p in reps),
       "total_tokens": sum(meta[p]["total_tokens"] for p in sel), "token_quantiles": q([meta[p]["total_tokens"] for p in sel]),
       "source_mix": collections.Counter(meta[p]["source"] for p in sel), "overlap_with": overlaps}
json.dump(man, open(f"{a.out}/lp15k_manifest.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in man.items() if k not in ("rows", "strata")}, indent=1, default=str))
