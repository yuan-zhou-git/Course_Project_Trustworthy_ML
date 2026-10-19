#!/usr/bin/env python
"""Short-trace selection (fourth data arm): 15K examples whose full conversation fits in --max-tokens (default 8K).

Motivation: the fine-tuned 3B models lose many samples to the 32K generation cap (unfinished reasoning), and small students
learn long teacher traces poorly (Li et al. 2025, "Small Models Struggle to Learn from Strong Reasoners"). This arm keeps
the same candidate pool as the learning-progress arm (math, balanced <think>, <= 16K tokens, dev panel and probe prompts
excluded) and only changes which lengths the student sees.
1. One response per prompt: the shortest one (most prompts have a single response).
2. Keep prompts whose chosen response gives <= --max-tokens total tokens.
3. Strata = source x total-token band (<=2K, 2-4K, 4-8K); quotas proportional to distinct prompts per stratum (same
   construction as the other selected arms); uniform random sample within a stratum (seeded).
Pre-registered: the rule is fixed before any benchmark result of the resulting model is seen.
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--lp-dir", required=True, help="dir with lp_candidates.json and lp_meta.json (shared candidate pool)")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=15000)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--bands", default="2048,4096")
ap.add_argument("--seed", type=int, default=20261018)
ap.add_argument("--manifests", nargs="*", default=[], help="other 15K manifests, to report overlap")
a = ap.parse_args()

meta = {int(k): v for k, v in json.load(open(f"{a.lp_dir}/lp_meta.json")).items()}
by_prompt = collections.defaultdict(list)
for p in meta:
    by_prompt[meta[p]["prompt_hash"]].append(p)
reps = sorted(min(ps, key=lambda p: (meta[p]["total_tokens"], p)) for ps in by_prompt.values())
elig = [p for p in reps if meta[p]["total_tokens"] <= a.max_tokens]
print(f"candidates {len(meta)}, distinct prompts {len(reps)}, prompts with shortest response <= {a.max_tokens} tokens: {len(elig)}")
assert len(elig) >= a.n, (len(elig), a.n)

edges = [int(x) for x in a.bands.split(",")] + [a.max_tokens]
def band(t):
    for k, e in enumerate(edges):
        if t <= e: return k
    return len(edges) - 1
strata = collections.defaultdict(list)
for p in elig:
    strata[(meta[p]["source"], band(meta[p]["total_tokens"]))].append(p)
keys = sorted(strata); E = len(elig)
raw = {k: a.n * len(strata[k]) / E for k in keys}
quota = {k: int(raw[k]) for k in keys}
for k in sorted(keys, key=lambda k: (-(raw[k] - quota[k]), k))[:a.n - sum(quota.values())]:
    quota[k] += 1
rng = random.Random(a.seed)
sel = []
for k in keys:
    sel += rng.sample(sorted(strata[k]), quota[k])
sel = sorted(sel)
assert len(sel) == a.n and len(set(sel)) == a.n

pool = json.load(open(a.pool, encoding="utf-8"))
def alpaca(r):
    return {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"],
            "system": r.get("system", "Please reason step by step, and put your final answer within \\boxed{}.")}
json.dump([alpaca(pool[p]) for p in sel], open(f"{a.out}/short15k.json", "w", encoding="utf-8"), ensure_ascii=False)

def q(xs):
    xs = sorted(xs); n = len(xs)
    return {str(t): xs[min(n - 1, int(t * n))] for t in (0.05, 0.25, 0.5, 0.75, 0.95)} if n else {}
overlaps = {}
for m in a.manifests:
    rows = {r["pos"] for r in json.load(open(m))["rows"]}
    overlaps[m.split("/")[-1]] = len(rows & set(sel))
man = {"method": f"short traces: shortest response per prompt, total tokens <= {a.max_tokens}; stratified by source x token band "
                 f"{edges} with quotas proportional to distinct prompts; uniform random within stratum (seed {a.seed})",
       "n": len(sel), "seed": a.seed, "rows": [{"pos": p, "index": None} for p in sel],
       "strata": [{"source": k[0], "band": k[1], "size": len(strata[k]), "quota": quota[k]} for k in keys],
       "n_prompts_eligible": len(elig), "total_tokens": sum(meta[p]["total_tokens"] for p in sel),
       "token_quantiles": q([meta[p]["total_tokens"] for p in sel]),
       "source_mix": collections.Counter(meta[p]["source"] for p in sel), "overlap_with": overlaps}
json.dump(man, open(f"{a.out}/short15k_manifest.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in man.items() if k not in ("rows", "strata")}, indent=1, default=str))
