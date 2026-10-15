#!/usr/bin/env python
"""Build the random 15K and the advanced (math, complete, length-stratified) 15K training sets.

Runs inside a Slurm job (CPU work only, but ARC requires a GPU allocation).
Everything is deterministic given --seed and the frozen pool file.

Outputs (in --out):
  pool_stats.json          schema audit, category/source counts, token quantiles, sha256 of the pool
  random15k.json           LLaMA-Factory alpaca-format list (instruction/input/output/system)
  random15k_manifest.json  selected row ids, seed, counts
  advanced15k.json
  advanced15k_manifest.json
  overlap_report.json      exact prompt matches between the pool and the eval benchmarks
"""
import argparse, collections, hashlib, json, os, random, re, sys, time

p = argparse.ArgumentParser()
p.add_argument("--pool", required=True)
p.add_argument("--eval-data", required=True, help="eval/data folder of the course repo")
p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=15000)
p.add_argument("--seed", type=int, default=20261015)
p.add_argument("--cutoff", type=int, default=16384)
p.add_argument("--bands", default="4096,8192", help="band edges in total tokens; last band ends at --cutoff")
p.add_argument("--category-field", default="category")
p.add_argument("--math-values", default="math", help="comma list of category values treated as math")
p.add_argument("--source-field", default="source")
p.add_argument("--skip-advanced", action="store_true")
p.add_argument("--fake-tokenizer", action="store_true", help="TEST ONLY: count tokens as chars/4 (no transformers needed)")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()

def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---------------------------------------------------------------- load + hash
h = hashlib.sha256()
with open(args.pool, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
pool_sha = h.hexdigest()
log("pool sha256", pool_sha)
with open(args.pool, encoding="utf-8") as f:
    pool = json.load(f)
N = len(pool)
log("rows", N)

# ---------------------------------------------------------------- schema audit
top_keys = collections.Counter()
extra_keys = collections.Counter()
for r in pool:
    top_keys.update(r.keys())
    if isinstance(r.get("extra_info"), dict):
        extra_keys.update(r["extra_info"].keys())
log("top-level keys", dict(top_keys))
log("extra_info keys", dict(extra_keys))

def extra(r, k):
    e = r.get("extra_info")
    return e.get(k) if isinstance(e, dict) else None

cat_counts = collections.Counter(str(extra(r, args.category_field)) for r in pool)
src_counts = collections.Counter(str(extra(r, args.source_field)) for r in pool)
log("category counts", cat_counts.most_common(20))
log("source counts", src_counts.most_common(20))
systems = collections.Counter(r.get("system", "") for r in pool)
log("system prompts", systems.most_common(3))
inputs_nonempty = sum(1 for r in pool if r.get("input"))
log("rows with non-empty input", inputs_nonempty)

# original row id: extra_info.index if present, else position
def row_id(i, r):
    idx = extra(r, "index")
    return {"pos": i, "index": idx}

# ---------------------------------------------------------------- random 15K (required baseline, unfiltered)
rng = random.Random(args.seed)
rand_pos = sorted(rng.sample(range(N), args.n))
def to_alpaca(r):
    return {"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"],
            "system": r.get("system", "Please reason step by step, and put your final answer within \\boxed{}.")}
with open(os.path.join(args.out, "random15k.json"), "w", encoding="utf-8") as f:
    json.dump([to_alpaca(pool[i]) for i in rand_pos], f, ensure_ascii=False)
json.dump({"seed": args.seed, "n": args.n, "pool_sha256": pool_sha, "pool_rows": N,
           "rows": [row_id(i, pool[i]) for i in rand_pos],
           "category_mix": collections.Counter(str(extra(pool[i], args.category_field)) for i in rand_pos),
           "source_mix": collections.Counter(str(extra(pool[i], args.source_field)) for i in rand_pos)},
          open(os.path.join(args.out, "random15k_manifest.json"), "w"), indent=1)
log("random15k written")

# ---------------------------------------------------------------- tokenize full conversations
if args.fake_tokenizer:
    class _FakeTok:
        def apply_chat_template(self, convs, tokenize=False, add_generation_prompt=False):
            return ["".join(m["content"] for m in c) for c in convs]
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[0] * (len(t) // 4 + 8) for t in texts]}
    tok = _FakeTok()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
def conv(r):
    return [{"role": "system", "content": r.get("system", "")},
            {"role": "user", "content": r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")},
            {"role": "assistant", "content": r["output"]}]
total_len = [0] * N
prompt_len = [0] * N
B = 256
for s in range(0, N, B):
    batch = pool[s:s + B]
    texts = tok.apply_chat_template([conv(r) for r in batch], tokenize=False)
    ptexts = tok.apply_chat_template([conv(r)[:2] for r in batch], tokenize=False, add_generation_prompt=True)
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    penc = tok(ptexts, add_special_tokens=False)["input_ids"]
    for j in range(len(batch)):
        total_len[s + j] = len(enc[j]); prompt_len[s + j] = len(penc[j])
    if (s // B) % 40 == 0:
        log("tokenized", s + len(batch))
def quant(xs):
    xs = sorted(xs); n = len(xs)
    return {q: xs[min(n - 1, int(q * n))] for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)} if n else {}
log("total token quantiles", quant(total_len))
log("prompt token quantiles", quant(prompt_len))
over = sum(1 for x in total_len if x > args.cutoff)
log(f"rows over cutoff {args.cutoff}:", over)

# ---------------------------------------------------------------- dedup + benchmark overlap
def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())
def hsh(s):
    return hashlib.md5(norm(s).encode("utf-8")).hexdigest()
prompt_hash = [hsh(r["instruction"]) for r in pool]
dup_counter = collections.Counter(prompt_hash)
first_of = {}
is_dup = [False] * N
for i, hv in enumerate(prompt_hash):
    if hv in first_of:
        is_dup[i] = True
    else:
        first_of[hv] = i
log("duplicate prompts (beyond first occurrence)", sum(is_dup))

bench_hashes = {}
for name in sorted(os.listdir(args.eval_data)):
    fp = os.path.join(args.eval_data, name, "test.jsonl")
    if not os.path.isfile(fp):
        continue
    for line in open(fp, encoding="utf-8"):
        d = json.loads(line)
        q = d.get("question") or d.get("problem") or d.get("Question") or ""
        if q:
            bench_hashes.setdefault(hsh(q), []).append(name)
overlap = [{"pos": i, "benchmarks": bench_hashes[hv]} for i, hv in enumerate(prompt_hash) if hv in bench_hashes]
rand_set = set(rand_pos)
json.dump({"method": "exact match of whitespace-normalized lowercased prompt text", "pool_overlap_rows": overlap,
           "random15k_overlap_rows": [o for o in overlap if o["pos"] in rand_set]},
          open(os.path.join(args.out, "overlap_report.json"), "w"), indent=1)
log("benchmark overlap rows in pool", len(overlap), "in random15k", sum(1 for o in overlap if o["pos"] in rand_set))

stats = {"pool_sha256": pool_sha, "rows": N, "top_keys": top_keys, "extra_info_keys": extra_keys,
         "category_counts": cat_counts, "source_counts": src_counts, "nonempty_input": inputs_nonempty,
         "total_tokens_quantiles": quant(total_len), "prompt_tokens_quantiles": quant(prompt_len),
         "rows_over_cutoff": over, "duplicate_prompts": sum(is_dup), "benchmark_overlap_rows": len(overlap),
         "random15k_total_tokens": sum(total_len[i] for i in rand_pos),
         "random15k_supervised_tokens": sum(total_len[i] - prompt_len[i] for i in rand_pos),
         "random15k_rows_over_cutoff": sum(1 for i in rand_pos if total_len[i] > args.cutoff)}
json.dump(stats, open(os.path.join(args.out, "pool_stats.json"), "w"), indent=1, default=str)
if args.skip_advanced:
    log("stats + random done; --skip-advanced set"); sys.exit(0)

# ---------------------------------------------------------------- advanced: math, complete, fits, stratified
math_vals = set(v.strip().lower() for v in args.math_values.split(","))
if not any(str(c).lower() in math_vals for c in cat_counts):
    log(f"ERROR: none of the category values {list(cat_counts)} match --math-values {math_vals}; aborting advanced selection")
    sys.exit(2)
think_open = re.compile(r"<think>"); think_close = re.compile(r"</think>")
reasons = collections.Counter()
eligible = []
for i, r in enumerate(pool):
    c = str(extra(r, args.category_field)).lower()
    out = r.get("output") or ""
    if c not in math_vals:
        reasons["not_math"] += 1; continue
    if is_dup[i]:
        reasons["duplicate_prompt"] += 1; continue
    if not out.strip():
        reasons["empty_output"] += 1; continue
    if len(think_open.findall(out)) != len(think_close.findall(out)):
        reasons["unbalanced_think"] += 1; continue
    if total_len[i] > args.cutoff:
        reasons["over_cutoff"] += 1; continue
    eligible.append(i)
log("advanced filter reasons", dict(reasons), "eligible", len(eligible))
if len(eligible) < args.n:
    log(f"ERROR: only {len(eligible)} eligible unique rows < {args.n}; stopping as pre-registered")
    sys.exit(3)

edges = [int(x) for x in args.bands.split(",")] + [args.cutoff]
def band(x):
    for k, e in enumerate(edges):
        if x <= e:
            return k
    return len(edges) - 1
strata = collections.defaultdict(list)
for i in eligible:
    strata[(str(extra(pool[i], args.source_field)), band(total_len[i]))].append(i)
keys = sorted(strata)
sizes = {k: len(strata[k]) for k in keys}
E = len(eligible)
# proportional quotas, largest remainder, capped by stratum size, deterministic redistribution
raw = {k: args.n * sizes[k] / E for k in keys}
quota = {k: int(raw[k]) for k in keys}
rem = args.n - sum(quota.values())
for k in sorted(keys, key=lambda k: (-(raw[k] - quota[k]), k))[:rem]:
    quota[k] += 1
while True:
    overflow = sum(max(0, quota[k] - sizes[k]) for k in keys)
    if overflow == 0:
        break
    for k in keys:
        quota[k] = min(quota[k], sizes[k])
    spare = [k for k in keys if quota[k] < sizes[k]]
    for k in sorted(spare, key=lambda k: (-(sizes[k] - quota[k]), k))[:overflow] if overflow <= len(spare) else spare:
        quota[k] += 1
    if sum(quota.values()) < args.n:
        need = args.n - sum(quota.values())
        for k in sorted(spare, key=lambda k: (-(sizes[k] - quota[k]), k)):
            if need == 0: break
            add = min(need, sizes[k] - quota[k]); quota[k] += add; need -= add
rng2 = random.Random(args.seed + 1)
adv_pos = []
for k in keys:
    adv_pos += sorted(rng2.sample(strata[k], quota[k]))
adv_pos = sorted(adv_pos)
assert len(adv_pos) == args.n and len(set(adv_pos)) == args.n
with open(os.path.join(args.out, "advanced15k.json"), "w", encoding="utf-8") as f:
    json.dump([to_alpaca(pool[i]) for i in adv_pos], f, ensure_ascii=False)
json.dump({"seed": args.seed, "n": args.n, "pool_sha256": pool_sha, "cutoff": args.cutoff, "band_edges": edges,
           "filters": {"category_in": sorted(math_vals), "dedup": "exact normalized prompt", "nonempty_output": True,
                       "balanced_think_tags": True, "total_tokens_le_cutoff": True},
           "filter_reasons": reasons, "eligible": E,
           "strata": [{"source": k[0], "band": k[1], "size": sizes[k], "quota": quota[k]} for k in keys],
           "rows": [row_id(i, pool[i]) for i in adv_pos],
           "total_tokens": sum(total_len[i] for i in adv_pos),
           "supervised_tokens": sum(total_len[i] - prompt_len[i] for i in adv_pos),
           "total_tokens_quantiles": quant([total_len[i] for i in adv_pos]),
           "source_mix": collections.Counter(str(extra(pool[i], args.source_field)) for i in adv_pos),
           "overlap_with_random15k": len(set(adv_pos) & rand_set),
           "benchmark_overlap_rows": [o for o in overlap if o["pos"] in set(adv_pos)]},
          open(os.path.join(args.out, "advanced15k_manifest.json"), "w"), indent=1, default=str)
log("advanced15k written; total tokens", sum(total_len[i] for i in adv_pos), "random15k total tokens", stats["random15k_total_tokens"])
log("DONE")
