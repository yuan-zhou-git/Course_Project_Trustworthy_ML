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
import argparse, glob, json, os, random, statistics, sys

LEADER = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
ALL = ["aime", "math", "cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="label=run_id ...")
ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = dict(r.split("=", 1) for r in a.runs)

def load(run_id, bench):
    files = [f for f in glob.glob(f"{a.root}/{run_id}/**/{bench}/test_*.jsonl", recursive=True) if "/completions/" not in f.replace("\\", "/")]
    if not files:
        return None
    rows = [json.loads(l) for l in open(files[0], encoding="utf-8")]
    rows.sort(key=lambda r: r.get("id", 0))
    return [[bool(x) for x in r["answers_correctness"]] for r in rows]

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
open(os.path.join(a.out, "results.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("\n".join(md))
