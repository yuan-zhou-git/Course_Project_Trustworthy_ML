#!/usr/bin/env python
"""Aggregate evaluation runs into report tables with paired bootstrap uncertainty.

Usage:
  python aggregate_results.py --root <eval_outputs dir> --runs base=base_v1 random=random15k_v1 advanced=advanced15k_v1 \
                              --out <reports dir> [--B 2000] [--seed 0]

Reads <root>/<run_id>/**/<bench>/test_*.jsonl (per-question `answers_correctness`, 8 bools) and <run_id>/summary.tsv.
Writes results.md and results.json. Stdlib only (safe on a login node).

Metrics per benchmark: pass@1 = mean over questions of mean(correct samples); pass@8 = mean over questions of any(correct).
Macro average ("leaderboard-style") = mean over the 6 locally available leaderboard benchmarks
(cn_math_2024, kaoyan, amc, minerva, olympiadbench, gpqa); AIME25 is not available locally, AIME24 and MATH are excluded
by the course leaderboard definition. Paired bootstrap resamples question ids within each benchmark (keeping each
question's 8 samples together) and recomputes the difference between runs; CI = 2.5/97.5 percentiles.
"""
import argparse, glob, json, os, random, re, statistics, sys

LEADER = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
ALL = ["aime", "math", "cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="label=run_id ...")
ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-data", default="", help="course eval/data dir; enables question-identity checks")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = dict(r.split("=", 1) for r in a.runs)

SIZES = {"aime": 30, "amc": 40, "cn_math_2024": 30, "gpqa": 198, "kaoyan": 199, "math": 500, "minerva": 272, "olympiadbench": 675}

def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())

BENCH_Q = {}
def bench_questions(bench):
    """Question texts of the benchmark in file order (for identity checks); None if --eval-data not given."""
    if not a.eval_data:
        return None
    if bench not in BENCH_Q:
        fp = os.path.join(a.eval_data, bench, "test.jsonl")
        BENCH_Q[bench] = [json.loads(l) for l in open(fp, encoding="utf-8")] if os.path.isfile(fp) else None
    return BENCH_Q[bench]

SHARD_SOURCES = {}
def load(run_id, bench):
    """Merge result files of a benchmark (a full run or index shards) with an explicit, score-independent rule:
    every question index is taken from the NARROWEST shard that covers it (ties: earliest start, then filename).
    Records carry no stable id in every benchmark, so the key is shard start + line position, and each record's question
    text is checked against the benchmark file at that index when --eval-data is given."""
    files = [f for f in glob.glob(f"{a.root}/{run_id}/**/{bench}/test_*.jsonl", recursive=True) if "/completions/" not in f.replace("\\", "/")]
    if not files:
        return None
    n_expected = SIZES.get(bench)
    shards = []
    for f in files:
        m = re.search(r"_s(\d+)_e(-?\d+)\.jsonl$", os.path.basename(f))
        start = int(m.group(1)) if m else 0
        end = int(m.group(2)) if m else -1
        if end == -1: end = n_expected
        shards.append((end - start, start, os.path.basename(f), f))
    shards.sort()
    owner = {}
    for width, start, name, f in shards:
        for idx in range(start, start + width):
            owner.setdefault(idx, (start, f))
    recs, used = {}, {}
    for width, start, name, f in shards:
        lines = [l for l in open(f, encoding="utf-8").read().split("\n") if l.strip()]   # not splitlines(): U+2028 inside strings
        if len(lines) != width:
            print(f"WARNING {run_id}/{bench}: {name} has {len(lines)} records, expected {width}", file=sys.stderr)
        for i, l in enumerate(lines):
            idx = start + i
            if owner.get(idx, (None, None))[1] != f:
                continue
            recs[idx] = json.loads(l); used[name] = used.get(name, 0) + 1
    SHARD_SOURCES[(run_id, bench)] = used
    idxs = sorted(recs)
    if n_expected is not None and (len(idxs) != n_expected or idxs != list(range(n_expected))):
        print(f"WARNING {run_id}/{bench}: coverage {len(idxs)}/{n_expected} (gaps: {[i for i in range(n_expected) if i not in recs][:5]})", file=sys.stderr)
    bq = bench_questions(bench)
    if bq is not None:
        bad = 0
        for idx in idxs:
            qref = bq[idx].get("question") or bq[idx].get("problem") or bq[idx].get("Question") or ""
            if _norm(qref) and _norm(qref) not in _norm(recs[idx].get("question", "")) and _norm(recs[idx].get("question", "")) not in _norm(qref):
                bad += 1
        if bad:
            print(f"WARNING {run_id}/{bench}: {bad} records whose question text does not match the benchmark file at that index", file=sys.stderr)
        else:
            print(f"ok {run_id}/{bench}: {len(idxs)} questions, identities verified, sources {used}", file=sys.stderr)
    return [[bool(x) for x in recs[i]["answers_correctness"]] for i in idxs]

def p1(c):  return statistics.mean(statistics.mean(q) for q in c)
def p8(c):  return statistics.mean(1.0 if any(q) else 0.0 for q in c)

data = {lab: {b: load(rid, b) for b in ALL} for lab, rid in runs.items()}
summ = {}
for lab, rid in runs.items():
    fp = f"{a.root}/{rid}/summary.tsv"
    if os.path.isfile(fp):
        hdr, *rows = [l.rstrip("\n").split("\t") for l in open(fp)]
        summ[lab] = {r[0]: dict(zip(hdr, r)) for r in rows}
    else:
        summ[lab] = {}

res = {"runs": runs, "per_benchmark": {}, "macro": {}, "paired": {}}
md = [f"# Results\n", f"Runs: {runs}\n"]
md.append("## Per-benchmark scores (pass@8 / pass@1)\n")
md.append("| benchmark | n | " + " | ".join(f"{lab} pass@8 | {lab} pass@1" for lab in runs) + " |")
md.append("|---|---:|" + "---:|---:|" * len(runs))
for b in ALL:
    cells, n = [], None
    for lab in runs:
        c = data[lab][b]
        if c is None:
            cells += ["-", "-"]; continue
        n = len(c)
        res["per_benchmark"].setdefault(b, {})[lab] = {"n": n, "pass8": p8(c), "pass1": p1(c)}
        cells += [f"{p8(c):.3f}", f"{p1(c):.3f}"]
    md.append(f"| {b}{'*' if b in LEADER else ''} | {n or '-'} | " + " | ".join(cells) + " |")
md.append("\n`*` = counted in the leaderboard-style macro average (AIME25 not available locally; AIME24 and MATH excluded by the course definition).\n")

md.append("## Macro average over the 6 local leaderboard benchmarks\n")
md.append("| run | macro pass@8 | macro pass@1 | benchmarks present |")
md.append("|---|---:|---:|---:|")
for lab in runs:
    present = [b for b in LEADER if data[lab][b] is not None]
    if present:
        m8 = statistics.mean(p8(data[lab][b]) for b in present); m1 = statistics.mean(p1(data[lab][b]) for b in present)
        res["macro"][lab] = {"pass8": m8, "pass1": m1, "n_bench": len(present)}
        md.append(f"| {lab} | {m8:.4f} | {m1:.4f} | {len(present)}/6 |")

md.append("\n## Generation length and cost (from summary.tsv)\n")
md.append("| run | benchmark | mean gen tokens | p95 | frac at length cap | elapsed s |")
md.append("|---|---|---:|---:|---:|---:|")
for lab in runs:
    for b in ALL:
        s = summ[lab].get(b)
        if s:
            md.append(f"| {lab} | {b} | {s['mean_gen_tokens']} | {s['p95_gen_tokens']} | {s['frac_len_capped']} | {s['elapsed_s']} |")

labels = list(runs)
pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i < j]
rng = random.Random(a.seed)
md.append(f"\n## Paired bootstrap over question ids (B={a.B}): difference = second minus first\n")
md.append("| pair | benchmark | d pass@8 | 95% CI | d pass@1 | 95% CI |")
md.append("|---|---|---:|---|---:|---|")
for x, y in pairs:
    macro8, macro1 = [], []
    common = [b for b in LEADER if data[x][b] is not None and data[y][b] is not None and len(data[x][b]) == len(data[y][b])]
    boot_macro = []
    per_b = {}
    for b in ALL:
        cx, cy = data[x][b], data[y][b]
        if cx is None or cy is None or len(cx) != len(cy):
            continue
        n = len(cx)
        d8 = p8(cy) - p8(cx); d1 = p1(cy) - p1(cx)
        b8, b1 = [], []
        for _ in range(a.B):
            idx = [rng.randrange(n) for _ in range(n)]
            sx = [cx[i] for i in idx]; sy = [cy[i] for i in idx]
            b8.append(p8(sy) - p8(sx)); b1.append(p1(sy) - p1(sx))
        b8.sort(); b1.sort()
        lo8, hi8 = b8[int(0.025 * a.B)], b8[int(0.975 * a.B) - 1]
        lo1, hi1 = b1[int(0.025 * a.B)], b1[int(0.975 * a.B) - 1]
        per_b[b] = {"d_pass8": d8, "ci_pass8": [lo8, hi8], "d_pass1": d1, "ci_pass1": [lo1, hi1], "boot8": b8, "boot1": b1}
        md.append(f"| {x} -> {y} | {b} | {d8:+.3f} | [{lo8:+.3f}, {hi8:+.3f}] | {d1:+.3f} | [{lo1:+.3f}, {hi1:+.3f}] |")
    if common:
        # macro bootstrap: independent resampling within each benchmark, then average the per-benchmark differences
        rng2 = random.Random(a.seed + 1)
        m8, m1 = [], []
        for _ in range(a.B):
            acc8, acc1 = [], []
            for b in common:
                cx, cy = data[x][b], data[y][b]; n = len(cx)
                idx = [rng2.randrange(n) for _ in range(n)]
                sx = [cx[i] for i in idx]; sy = [cy[i] for i in idx]
                acc8.append(p8(sy) - p8(sx)); acc1.append(p1(sy) - p1(sx))
            m8.append(statistics.mean(acc8)); m1.append(statistics.mean(acc1))
        m8.sort(); m1.sort()
        d8 = statistics.mean(p8(data[y][b]) - p8(data[x][b]) for b in common)
        d1 = statistics.mean(p1(data[y][b]) - p1(data[x][b]) for b in common)
        lo8, hi8 = m8[int(0.025 * a.B)], m8[int(0.975 * a.B) - 1]; lo1, hi1 = m1[int(0.025 * a.B)], m1[int(0.975 * a.B) - 1]
        md.append(f"| {x} -> {y} | **macro ({len(common)} benchmarks)** | **{d8:+.4f}** | [{lo8:+.4f}, {hi8:+.4f}] | **{d1:+.4f}** | [{lo1:+.4f}, {hi1:+.4f}] |")
        per_b["macro"] = {"d_pass8": d8, "ci_pass8": [lo8, hi8], "d_pass1": d1, "ci_pass1": [lo1, hi1], "benchmarks": common}
    for b in per_b:
        per_b[b].pop("boot8", None); per_b[b].pop("boot1", None)
    res["paired"][f"{x}->{y}"] = per_b

md.append("\nThe bootstrap captures question-sampling uncertainty for these specific trained models only; it does not capture training-seed or data-draw variance (one training run per arm).\n")
md.append("\n## Shard sources used per (run, benchmark)\n")
md.append("Rule: each question index is taken from the narrowest result shard covering it (ties: earliest start). Counts = questions taken from each file.\n")
for (rid, b), used in sorted(SHARD_SOURCES.items()):
    if len(used) > 1:
        md.append(f"- {rid}/{b}: " + ", ".join(f"`{k}`: {v}" for k, v in sorted(used.items())))
open(os.path.join(a.out, "results.md"), "w", encoding="utf-8").write("\n".join(md))
res["shard_sources"] = {f"{k[0]}/{k[1]}": v for k, v in SHARD_SOURCES.items()}
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
# compact per-question correctness so the bootstrap is reproducible from public files
pq = {lab: {b: ["".join("1" if x else "0" for x in q) for q in data[lab][b]] for b in ALL if data[lab][b] is not None} for lab in runs}
json.dump({"format": "per question: 8-char string of 0/1 per sample, in benchmark file order (index = line in eval/data/<bench>/test.jsonl)",
           "runs": runs, "correctness": pq}, open(os.path.join(a.out, "per_question_correctness.json"), "w"))
print("\n".join(md))
