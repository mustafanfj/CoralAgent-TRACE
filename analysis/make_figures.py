"""Revision figures (static, for the manuscript). Numbers are recomputed from results/v13, not typed in.

fig_paired_differences.(png|pdf): TRACE minus each comparator, paired image-cluster bootstrap 95% CI.
fig_selective_accuracy.(png|pdf): accuracy vs coverage when answers with the least trusted evidence are referred.
"""
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


def sc(r):
    return float(V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"]) >= 0.5)


A3 = load("A3v13_trace")
H = [k for k, r in A3.items() if r["split"] == "H"]
D = [k for k, r in A3.items() if r["split"] == "D"]
E6 = load("E6_self_consistency")
H100 = [k for k in E6 if k in A3 and A3[k]["split"] == "H"]

comparisons = [  # (group, label, method file, keys); difference = TRACE minus comparator
    ("Held-out H (2,400 questions)", "Strict GPT-5 (primary)", "B0b_strict_v13", H),
    ("Held-out H (2,400 questions)", "GPT-4.1", "B1_gpt41_v13", H),
    ("Held-out H (2,400 questions)", "Grok-4.3", "B2_grok_v13", H),
    ("Matched controls, H100 (600 questions)", "Generic GPT-5 agent, same tools", "E7a_generic_gpt5_agent", H100),
    ("Matched controls, H100 (600 questions)", "Single GPT-5 call, same information", "E5_single_call_same_info", H100),
    ("Matched controls, H100 (600 questions)", "GPT-5 self-consistency, k = 3", "E6_self_consistency", H100),
]
rows = []
for g, lab, m, keys in comparisons:
    d = load(m)
    ks = [k for k in keys if k in d]
    bs = E.paired_bootstrap([(A3[k]["image_id"], sc(A3[k]), sc(d[k])) for k in ks])
    rows.append((g, lab, 100 * bs["diff"], 100 * bs["ci_low"], 100 * bs["ci_high"]))

fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
y, ticks, labels, last = 0, [], [], None
for g, lab, d, lo, hi in rows:
    if g != last:
        if last is not None:
            y += 0.6
        ax.text(-9.6, y, g, fontsize=8.5, color=MUTED, va="center", fontweight="bold")
        y += 1
        last = g
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot([d], [y], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
    ax.text(hi + 0.4, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK)
    ticks.append(y)
    labels.append(lab)
    y += 1
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(ticks, labels)
ax.invert_yaxis()
ax.set_xlim(-10, 12)
ax.set_xlabel("Accuracy difference, CoralAgent-TRACE minus comparator (percentage points, 95% image-cluster CI)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_paired_differences.{ext}"), bbox_inches="tight")
plt.close(fig)

# selective accuracy
ja, je, e7 = load("E8_judge_A3v13"), load("E8_judge_E7a"), load("E7a_generic_gpt5_agent")
keys = sorted(k for k in ja if k in je)


def curve(j, s):
    rk = sorted(keys, key=lambda k: (bool(j[k]["rating"].get("contradicts_answer")), -float(j[k]["rating"]["overall"]), k))
    xs, ys = [], []
    for i in range(60, len(rk) + 1, 6):
        xs.append(100 * i / len(rk))
        ys.append(100 * sum(sc(s[k]) for k in rk[:i]) / i)
    return xs, ys


fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=200)
for (j, s, name, col) in [(ja, A3, "CoralAgent-TRACE", BLUE), (je, e7, "Generic GPT-5 agent", ORANGE)]:
    xs, ys = curve(j, s)
    ax.plot(xs, ys, color=col, lw=2, label=name)
    ax.text(xs[-1] + 1, ys[-1], name, color=INK, fontsize=8, va="center")
ax.set_xlim(10, 125)
ax.set_xticks([20, 40, 60, 80, 100])
ax.set_xlabel("Answers accepted (%); the rest are referred to an expert")
ax.set_ylabel("Accuracy of accepted answers (%)")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_selective_accuracy.{ext}"), bbox_inches="tight")
print("figures written to", OUT)
