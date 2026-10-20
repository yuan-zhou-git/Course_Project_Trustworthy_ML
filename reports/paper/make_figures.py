#!/usr/bin/env python
"""Figure for the paper: share of samples cut off at the 32,768-token limit on the six leaderboard benchmarks.

Reads reports/five_models/stopping_failures.json; writes fig_cutoff.pdf next to this script.
Usage: python make_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "..", "five_models", "stopping_failures.json")))

BENCH = [("cn_math_2024", "CN-Math"), ("kaoyan", "Kaoyan"), ("amc", "AMC"), ("minerva", "Minerva"),
         ("olympiadbench", "Olympiad"), ("gpqa", "GPQA")]
# fixed categorical order (reference palette slots 1-3); identity also via legend and hatching for grayscale print
SERIES = [("base", "Base", "#2a78d6", None), ("random", "Random 15K", "#eb6834", None),
          ("ep5", "Final model", "#1baf7a", "////")]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "hatch.linewidth": 0.6, "hatch.color": "#0f6b4a", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 1.75))
w = 0.26
for k, (key, label, color, hatch) in enumerate(SERIES):
    xs = [i + (k - 1) * w for i in range(len(BENCH))]
    ys = [100.0 * data[key][b]["capped"] / data[key][b]["samples"] for b, _ in BENCH]
    ax.bar(xs, ys, width=w, color=color, edgecolor="white", linewidth=0.6, hatch=hatch, label=label, zorder=3)
ax.set_xticks(range(len(BENCH)))
ax.set_xticklabels([n for _, n in BENCH])
ax.set_ylabel("Truncation rate (%)")
ax.set_ylim(0, 30)
ax.set_yticks([0, 10, 20, 30])
ax.grid(axis="y", color="#e2e1dc", linewidth=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#8c8b86"); ax.spines[s].set_linewidth(0.6)
ax.tick_params(colors="#3a3936", width=0.6, length=2)
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1.2, columnspacing=1.2)
fig.tight_layout(pad=0.2)
out = os.path.join(HERE, "fig_cutoff.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)
