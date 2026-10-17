#!/usr/bin/env python
"""Learning-progress selection, step 1: probe training set + candidate list.

Eligible rows: math category, non-empty output with balanced <think> tags, full conversation <= cutoff tokens under the
Qwen chat template. Duplicated prompts are KEPT here (alternative responses to the same question are chosen among later,
GRAPE-style). Rows of the held-out dev panel are excluded so it stays a clean generalization check.
- probe set: N_PROBE random eligible rows (seed), written in alpaca format for a short probe SFT
- candidates: all other eligible rows (positions), with per-row metadata for stratification
"""
import argparse, collections, hashlib, json, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--dev-panel", default="")
ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--n-probe", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20261017)
ap.add_argument("--cutoff", type=int, default=16384)
a = ap.parse_args()

pool = json.load(open(a.pool, encoding="utf-8"))
dev_pos = set()
if a.dev_panel:
    dev_pos = {json.loads(l)["pos"] for l in open(a.dev_panel, encoding="utf-8")}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
think_o, think_c = re.compile(r"<think>"), re.compile(r"</think>")

elig, meta = [], {}
B = 256
rows = [(i, r) for i, r in enumerate(pool) if str((r.get("extra_info") or {}).get("category", "")).lower() == "math"
        and i not in dev_pos and (r.get("output") or "").strip()
        and len(think_o.findall(r["output"])) == len(think_c.findall(r["output"]))]
for s in range(0, len(rows), B):
    batch = rows[s:s + B]
    convs = [[{"role": "system", "content": r.get("system", "")}, {"role": "user", "content": r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")},
              {"role": "assistant", "content": r["output"]}] for _, r in batch]
    enc = tok(tok.apply_chat_template(convs, tokenize=False), add_special_tokens=False)["input_ids"]
    for (i, r), ids in zip(batch, enc):
        if len(ids) <= a.cutoff:
            elig.append(i)
            meta[i] = {"source": str((r.get("extra_info") or {}).get("source")), "total_tokens": len(ids),
                       "prompt_hash": hashlib.md5(norm(r["instruction"]).encode()).hexdigest()}
print("math rows", len(rows), "eligible (fit cutoff)", len(elig))
rng = random.Random(a.seed)
probe = sorted(rng.sample(elig, a.n_probe))
probe_set = set(probe)
probe_hashes = {meta[p]["prompt_hash"] for p in probe}
# candidates: eligible rows not in the probe set and not sharing a prompt with a probe row (no memorization leakage)
cands = [p for p in elig if p not in probe_set and meta[p]["prompt_hash"] not in probe_hashes]
def alpaca(r):
    return {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"],
            "system": r.get("system", "Please reason step by step, and put your final answer within \\boxed{}.")}
json.dump([alpaca(pool[p]) for p in probe], open(f"{a.out_dir}/probe2k.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(cands, open(f"{a.out_dir}/lp_candidates.json", "w"))
json.dump({str(p): meta[p] for p in cands}, open(f"{a.out_dir}/lp_meta.json", "w"))
json.dump({"seed": a.seed, "n_probe": len(probe), "probe_rows": probe, "n_candidates": len(cands), "n_eligible": len(elig),
           "dev_panel_excluded": len(dev_pos), "probe_tokens": sum(meta[p]["total_tokens"] for p in probe),
           "candidate_sources": collections.Counter(meta[p]["source"] for p in cands),
           "distinct_prompts_in_candidates": len({meta[p]["prompt_hash"] for p in cands})},
          open(f"{a.out_dir}/lp_build_manifest.json", "w"), indent=1)
print("probe", len(probe), "candidates", len(cands), "distinct prompts", len({meta[p]["prompt_hash"] for p in cands}),
      "candidate tokens", sum(meta[p]["total_tokens"] for p in cands))
