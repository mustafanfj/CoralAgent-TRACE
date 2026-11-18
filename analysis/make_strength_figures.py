"""Three result figures for the revision:

fig_bleaching_health    bleaching/health accuracy: (a) all models, 555 held-out questions; (b) GPT-5 controls, matched subset
fig_accuracy_cost       accuracy vs cost per question for the GPT-5-based methods (matched 600-question subset)
fig_error_exposure      error exposure of the evidence traces under both blinded judges

Accuracies and CIs are recomputed from results/v13. Judge rates and differences are the values in
analysis/E8_judge.md and analysis/E8b_agreement.md; costs are the batch costs in analysis/H100_controls.md.
"""
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")
OUT = os.path.join(HERE, "figures")
BLUE, GREY, INK, MUTED, GRID, SURFACE, LIGHTGREY = "#2a78d6", "#a8a7a2", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#cfcec9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


def sc(r):
    return float(V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"]) >= 0.5)


def acc_ci(d, keys, n_boot=4000, seed=2026):
    by = {}
    for k in keys:
        by.setdefault(d[k]["image_id"], []).append(sc(d[k]))
    s = np.array([sum(v) for v in by.values()])
    n = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def bars(ax, items, ylim, ylabel=None, fmt="{:.3f}"):
    for i, (name, a, lo, hi) in enumerate(items):
        ax.bar(i, a, width=0.62, color=BLUE if i == 0 else GREY, zorder=2)
        ax.plot([i, i], [lo, hi], color=INK, lw=1, zorder=3)
        ax.text(i, hi + 0.015, fmt.format(a), ha="center", fontsize=8, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(items)), [it[0] for it in items], fontsize=7.4)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


A3 = load("A3v13_trace")
B0b, B1, B2 = load("B0b_strict_v13"), load("B1_gpt41_v13"), load("B2_grok_v13")
E5, E6, E7 = load("E5_single_call_same_info"), load("E6_self_consistency"), load("E7a_generic_gpt5_agent")
H = [k for k, r in A3.items() if r["split"] == "H"]
H100 = [k for k in E6 if k in A3 and A3[k]["split"] == "H"]
BH = [k for k in H if A3[k]["question_type"] == "bleaching_health"]
BH100 = [k for k in H100 if A3[k]["question_type"] == "bleaching_health"]

# ------------------------------------------------------------------ 1. bleaching / health
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=220, gridspec_kw={"width_ratios": [4, 5]})
pa = [(n, *acc_ci(d, [k for k in BH if k in d])) for n, d in
      [("CoralAgent-\nTRACE", A3), ("Strict\nGPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]]
pb = [(n, *acc_ci(d, [k for k in BH100 if k in d])) for n, d in
      [("CoralAgent-\nTRACE", A3), ("Generic\nGPT-5 agent", E7), ("Single\nGPT-5 call,\nsame info", E5),
       ("GPT-5 self-\nconsistency", E6), ("Strict\nGPT-5", B0b)]]
bars(axes[0], pa, (0, 1.0), "Accuracy on bleaching/health questions")
bars(axes[1], pb, (0, 1.0))
axes[0].set_title(f"(a) Zero-shot models, {len(BH)} held-out questions", fontsize=8.5, loc="left")
axes[1].set_title(f"(b) GPT-5 controls, {len(BH100)} questions of the matched subset", fontsize=8.5, loc="left")
save(fig, "fig_bleaching_health")
print("bleaching (a):", [(n.replace(chr(10), ' '), round(a, 3), round(lo, 3), round(hi, 3)) for n, a, lo, hi in pa])
print("bleaching (b):", [(n.replace(chr(10), ' '), round(a, 3)) for n, a, lo, hi in pb])

# ------------------------------------------------------------------ 2. accuracy vs cost (matched subset)
cost = {"CoralAgent-TRACE": 0.0095, "Generic GPT-5 agent\n(same tools and verifier)": 0.0109,
        "Single GPT-5 call,\nsame information": 0.0020, "GPT-5 self-consistency (k = 3)": 0.0034, "Strict GPT-5": 0.0011}
sysm = dict(zip(cost, [A3, E7, E5, E6, B0b]))
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
offs = {"CoralAgent-TRACE": (8, 6), "Generic GPT-5 agent\n(same tools and verifier)": (-8, -22),
        "Single GPT-5 call,\nsame information": (8, -16), "GPT-5 self-consistency (k = 3)": (8, 4), "Strict GPT-5": (8, -10)}
for name, d in sysm.items():
    a, lo, hi = acc_ci(d, [k for k in H100 if k in d])
    main = name == "CoralAgent-TRACE"
    ax.plot([cost[name]] * 2, [100 * lo, 100 * hi], color=BLUE if main else GREY, lw=1.2, zorder=2)
    ax.plot(cost[name], 100 * a, "o", ms=9 if main else 7, color=BLUE if main else GREY, mec=SURFACE, mew=1.5, zorder=3)
    ha = "right" if offs[name][0] < 0 else "left"
    ax.annotate(f"{name}\n{a:.3f}", (cost[name], 100 * a), textcoords="offset points", xytext=offs[name], ha=ha,
                fontsize=7.5, fontweight="bold" if main else "normal", color=INK)
ax.annotate("", xy=(0.0095, 54.0), xytext=(0.0109, 50.6),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2, connectionstyle="arc3,rad=0.25"))
ax.set_xlim(0, 0.0135)
ax.set_ylim(44, 61)
ax.set_xlabel("Cost per question (USD, batch prices)")
ax.set_ylabel("Accuracy (%), 95% image-cluster CI")
ax.grid(color=GRID, lw=0.6, zorder=0)
save(fig, "fig_accuracy_cost")

# ------------------------------------------------------------------ 3. error exposure under both judges
panels = [("(a) Incorrect answers whose trace\nflags a contradiction", "higher is better",
           [("Grok-4.3\njudge", 0.564, 0.255, "+0.31 [+0.24, +0.38]"), ("GPT-4.1\njudge", 0.341, 0.104, "+0.24 [+0.19, +0.29]")]),
          ("(b) Incorrect answers backed by\nconvincing evidence", "lower is better",
           [("Grok-4.3\njudge", 0.337, 0.570, "-0.23 [-0.30, -0.17]"), ("GPT-4.1\njudge", 0.568, 0.809, "-0.24 [-0.31, -0.18]")]),
          ("(c) AUROC of the evidence rating\nfor answer correctness", "higher is better",
           [("Grok-4.3\njudge", 0.674, 0.590, "+0.08 [+0.03, +0.14]"), ("GPT-4.1\njudge", 0.639, 0.536, "+0.10 [+0.06, +0.15]")])]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), dpi=220)
for ax, (title, note, rows) in zip(axes, panels):
    w = 0.36
    for i, (judge, t, g, diff) in enumerate(rows):
        ax.bar(i - w / 2, t, width=w - 0.03, color=BLUE, zorder=2, label="CoralAgent-TRACE" if i == 0 else None)
        ax.bar(i + w / 2, g, width=w - 0.03, color=GREY, zorder=2, label="Generic GPT-5 agent" if i == 0 else None)
        ax.text(i - w / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(i + w / 2, g + 0.02, f"{g:.2f}", ha="center", fontsize=7.5)
        d0, d1 = diff.split(" ", 1)
        ax.text(i, 1.0, f"diff. {d0}\n{d1}", ha="center", fontsize=6.4, color=MUTED, va="bottom")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, fontsize=8, loc="left")
    ax.text(0.0, -0.3, note, transform=ax.transAxes, fontsize=7, color=MUTED, style="italic")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Rate or AUROC (600 held-out questions)")
axes[1].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
save(fig, "fig_error_exposure")
print("done")
