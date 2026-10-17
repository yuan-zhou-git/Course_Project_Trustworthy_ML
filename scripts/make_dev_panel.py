#!/usr/bin/env python
"""Build a held-out development panel for checkpoint screening (no benchmark questions involved).

Source: the 100K AceReason pool. Keep math rows that are in NEITHER 15K training set, whose teacher response ends with a
parseable \\boxed{...} answer (the reference is the teacher's answer, i.e. teacher-agreement, not independent ground truth),
and whose prompt is not an exact duplicate of a training prompt. Stratify by teacher-response length tertiles and sample
N questions with a fixed seed. Output: jsonl with question, answer, pos, index, len_chars.
"""
import argparse, collections, hashlib, json, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--manifests", nargs="+", required=True, help="random15k_manifest.json advanced15k_manifest.json")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=20261016)
a = ap.parse_args()

pool = json.load(open(a.pool, encoding="utf-8"))
used = set()
for m in a.manifests:
    used |= {r["pos"] for r in json.load(open(m))["rows"]}
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
train_prompt_hash = {hashlib.md5(norm(pool[p]["instruction"]).encode()).hexdigest() for p in used}

def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{": depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0: return s[j + 1:k]
        k += 1
    return None

cands = []
for pos, r in enumerate(pool):
    if pos in used: continue
    e = r.get("extra_info") or {}
    if str(e.get("category", "")).lower() != "math": continue
    if hashlib.md5(norm(r["instruction"]).encode()).hexdigest() in train_prompt_hash: continue
    out = r.get("output") or ""
    if "</think>" not in out: continue
    ans = last_boxed(out[out.rfind("</think>"):])          # answer must be in the conclusion, after the thinking block
    if not ans or len(ans) > 60 or "\\text" in ans and len(ans) > 30: continue
    cands.append((pos, e.get("index"), ans.strip(), len(out)))
print("candidates", len(cands))
cands.sort(key=lambda x: x[3])
t = len(cands) // 3
bands = [cands[:t], cands[t:2 * t], cands[2 * t:]]
rng = random.Random(a.seed)
per = [a.n // 3, a.n // 3, a.n - 2 * (a.n // 3)]
chosen = []
for b, k in zip(bands, per):
    chosen += rng.sample(b, k)
chosen.sort()
with open(a.out, "w", encoding="utf-8") as f:
    for pos, idx, ans, L in chosen:
        f.write(json.dumps({"question": pool[pos]["instruction"], "answer": ans, "pos": pos, "index": idx, "len_chars": L}, ensure_ascii=False) + "\n")
print("wrote", a.out, len(chosen), "questions; length bands (chars):", [(b[0][3], b[-1][3]) for b in bands],
      "sources:", collections.Counter(str((pool[p].get("extra_info") or {}).get("source")) for p, _, _, _ in chosen))
