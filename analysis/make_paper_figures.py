"""Manuscript figures for the revision, one per figure of the submitted version (numbers recomputed from results/v13).

fig_model_comparison      single-model comparison (submitted Fig. 2)
fig_question_type_heatmap accuracy by method and question type, matched 600-question study (submitted Fig. 3)
fig_ablation_accuracy     absolute accuracy of the full system and its component variants (submitted Fig. 4)
fig_component_impact      variant minus full system, paired 95% CI (submitted Fig. 5)
fig_scale_consistency     cumulative TRACE minus strict GPT-5 from 100 to 500 images (submitted Fig. 6)
fig_capability_profile    TRACE accuracy by question type on the 2,400 held-out questions (submitted Fig. 7)
fig_bleaching_profile     visible-bleaching operating profile (submitted Fig. 8)
"""
import csv
import glob
import gzip
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, LIGHT, ORANGE, AQUA, YELLOW = "#2a78d6", "#9cc3ef", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


_cache = {}


def sc(r):
    key = (r["sample_id"], r.get("method"), r.get("answer_pred"))
    if key not in _cache:
        _cache[key] = float(V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "",
                                                   r["question_type"]) >= 0.5)
    return _cache[key]


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def acc_ci(d, keys, n_boot=4000, seed=2026):
    by = {}
    for k in keys:
        by.setdefault(d[k]["image_id"], []).append(sc(d[k]))
    imgs = list(by)
    s = np.array([sum(by[i]) for i in imgs])
    n = np.array([len(by[i]) for i in imgs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(imgs), size=(n_boot, len(imgs)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return 100 * s.sum() / n.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


A3 = load("A3v13_trace")
H = sorted(k for k, r in A3.items() if r["split"] == "H")
D = sorted(k for k, r in A3.items() if r["split"] == "D")
E6 = load("E6_self_consistency")
H100 = sorted(k for k in E6 if k in A3 and A3[k]["split"] == "H")
B0b, B1, B2 = load("B0b_strict_v13"), load("B1_gpt41_v13"), load("B2_grok_v13")
E5, E7 = load("E5_single_call_same_info"), load("E7a_generic_gpt5_agent")
TYPES = [("bleaching_health", "Bleaching /\nhealth"), ("color_visual", "Colour"), ("genus_taxonomy", "Genus /\ntaxonomy"),
         ("morphology_visual", "Morphology"), ("presence_count", "Presence /\ncount"), ("other", "General")]

# ---------------------------------------------------------------- 1. model comparison (two panels)
panels = [("(a) Held-out set: 400 images, 2,400 questions", H,
           [("CoralAgent-\nTRACE", A3), ("Strict\nGPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]),
          ("(b) Matched controls: 100 held-out images, 600 questions", H100,
           [("CoralAgent-\nTRACE", A3), ("Generic\nGPT-5 agent,\nsame tools", E7), ("Single\nGPT-5 call,\nsame info", E5),
            ("GPT-5 self-\nconsistency\n(k = 3)", E6), ("Strict\nGPT-5", B0b)])]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=220, gridspec_kw={"width_ratios": [4, 5]})
for ax, (title, keys, systems) in zip(axes, panels):
    for i, (name, d) in enumerate(systems):
        ks = [k for k in keys if k in d]
        a, lo, hi = acc_ci(d, ks)
        col = BLUE if i == 0 else GREY
        ax.bar(i, a, width=0.62, color=col, zorder=2)
        ax.plot([i, i], [lo, hi], color=INK, lw=1, zorder=3)
        ax.text(i, hi + 1.2, f"{a / 100:.3f}", ha="center", fontsize=8, color=INK,
                fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(systems)), [s[0] for s in systems], fontsize=7.2, rotation=0)
    ax.set_ylim(0, 68)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Accuracy (%), 95% image-cluster CI")
for lab in axes[0].get_xticklabels() + axes[1].get_xticklabels():
    if lab.get_text().startswith("CoralAgent"):
        lab.set_fontweight("bold")
save(fig, "fig_model_comparison")

# ---------------------------------------------------------------- 2. question-type heatmap on the matched study
methods = [("CoralAgent-TRACE", A3), ("Generic GPT-5 agent", E7), ("Single GPT-5 call", E5),
           ("GPT-5 self-consistency", E6), ("Strict GPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]
M = np.zeros((len(methods), len(TYPES)))
ns = []
for j, (t, _) in enumerate(TYPES):
    ks = [k for k in H100 if A3[k]["question_type"] == t]
    ns.append(len(ks))
    for i, (_, d) in enumerate(methods):
        M[i, j] = 100 * np.mean([sc(d[k]) for k in ks])
fig, ax = plt.subplots(figsize=(7.0, 3.3), dpi=220)
im = ax.imshow(M, cmap="Blues", vmin=10, vmax=85, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        best = M[i, j] >= M[:, j].max() - 1e-9
        ax.text(j, i, f"{M[i, j] / 100:.2f}", ha="center", va="center", fontsize=8,
                color=SURFACE if M[i, j] > 55 else INK, fontweight="bold" if best else "normal")
ax.set_xticks(range(len(TYPES)), [f"{lab}\n(n = {n})" for (_, lab), n in zip(TYPES, ns)], fontsize=7.5)
ax.set_yticks(range(len(methods)), [m[0] for m in methods])
ax.get_yticklabels()[0].set_fontweight("bold")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Accuracy (%)", fontsize=8)
cb.outline.set_visible(False)
save(fig, "fig_question_type_heatmap")

# ---------------------------------------------------------------- 3. ablation accuracy and 4. component impact (development set)
abl = [("Full CoralAgent-TRACE", "A3v13_trace"), ("Without GUARD", "A5v13_no_guard"),
       ("Prior + SCOPE only", "A1v13_prior_scope"), ("Without SCOPE crop", "A3v13_no_scope"),
       ("All tools on every question", "A2v13_fixed_all_tools"), ("Without retrieval", "A4v13_no_retrieval"),
       ("Without prior", "A6v13_no_prior"), ("SCOPE only", "A0v13_scope_only")]
dd = {m: load(m) for _, m in abl}
fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=220)
for i, (lab, m) in enumerate(abl):
    a, lo, hi = acc_ci(dd[m], [k for k in D if k in dd[m]])
    ax.barh(i, a, height=0.62, color=BLUE if i == 0 else GREY, zorder=2)
    ax.plot([lo, hi], [i, i], color=INK, lw=1, zorder=3)
    ax.text(hi + 0.6, i, f"{a / 100:.3f}", va="center", fontsize=8, fontweight="bold" if i == 0 else "normal")
ax.set_yticks(range(len(abl)), [a[0] for a in abl])
ax.get_yticklabels()[0].set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 68)
ax.set_xlabel("Accuracy on the development set (%), 600 questions, 95% image-cluster CI")
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.tick_params(axis="y", length=0)
save(fig, "fig_ablation_accuracy")

rows = []
for lab, m in abl[1:]:
    ks = [k for k in D if k in dd[m]]
    bs = E.paired_bootstrap([(A3[k]["image_id"], sc(dd[m][k]), sc(A3[k])) for k in ks])
    rows.append((lab, 100 * bs["diff"], 100 * bs["ci_low"], 100 * bs["ci_high"]))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=220)
for i, (lab, d, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot([d], [i], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
    ax.text(hi + 0.4, i, f"{d:+.1f}", va="center", fontsize=8, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlabel("Change in accuracy when the component is removed or the policy changed\n"
              "(percentage points vs the full system, development set, 95% image-cluster CI)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.tick_params(axis="y", length=0)
save(fig, "fig_component_impact")

# ---------------------------------------------------------------- 5. scale consistency across the held-out images
bank = glob.glob(os.path.join(HERE, "..", "data_release", "*.csv")) or \
    glob.glob(os.path.join(HERE, "..", "..", "coralagent-trace", "data_release", "*.csv")) or \
    glob.glob(os.path.join(HERE, "..", "data", "outputs", "*bank500*.csv"))
idx = {r["sample_id"]: int(r["active_image_index"]) for r in csv.DictReader(open(bank[0], encoding="utf-8"))}
keys = [k for k in H if k in B0b]
by_img = {}
for k in keys:
    by_img.setdefault(idx[k], []).append(sc(A3[k]) - sc(B0b[k]))
order = sorted(by_img)                      # bank order: images 101 to 500
diff_sum = np.array([sum(by_img[i]) for i in order])
cnt = np.array([len(by_img[i]) for i in order])
rng = np.random.default_rng(2026)
xs, ys, los, his = [], [], [], []
for n in range(25, len(order) + 1, 25):
    s_, c_ = diff_sum[:n], cnt[:n]
    bi = rng.integers(0, n, size=(4000, n))
    b = s_[bi].sum(1) / c_[bi].sum(1)
    xs.append(n); ys.append(100 * s_.sum() / c_.sum())
    los.append(100 * np.percentile(b, 2.5)); his.append(100 * np.percentile(b, 97.5))
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=220)
ax.axhspan(-2, 2, color=GRID, alpha=0.5, lw=0, zorder=0)
ax.fill_between(xs, los, his, color=LIGHT, alpha=0.6, lw=0, zorder=1, label="95% image-cluster CI")
ax.plot(xs, ys, color=BLUE, lw=2, zorder=3, label="Cumulative accuracy difference")
ax.plot([xs[-1]], [ys[-1]], "D", color=BLUE, ms=6, mec=SURFACE, mew=1.2, zorder=4)
ax.text(xs[-1] + 6, ys[-1], f"{ys[-1]:+.1f} [{los[-1]:+.1f}, {his[-1]:+.1f}]", va="center", fontsize=8)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlim(15, 470)
ax.set_xticks([25, 100, 200, 300, 400])
ax.set_xlabel("Held-out images evaluated, in bank order (6 questions per image)")
ax.set_ylabel("CoralAgent-TRACE minus\nstrict GPT-5 (points)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
save(fig, "fig_scale_consistency")
print("scale:", [(x, round(y, 2), round(l, 2), round(h, 2)) for x, y, l, h in zip(xs, ys, los, his)][::4])

# ---------------------------------------------------------------- 6. capability profile on the held-out set
fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=220)
vals = []
for t, lab in TYPES:
    ks = [k for k in H if A3[k]["question_type"] == t]
    a, lo, hi = acc_ci(A3, ks)
    vals.append((lab.replace("\n", " "), a, lo, hi, len(ks)))
vals.sort(key=lambda v: -v[1])
for i, (lab, a, lo, hi, n) in enumerate(vals):
    ax.bar(i, a, width=0.62, color=BLUE, zorder=2)
    ax.plot([i, i], [lo, hi], color=INK, lw=1, zorder=3)
    ax.text(i, hi + 1.5, f"{a / 100:.3f}", ha="center", fontsize=8)
    ax.text(i, 4, f"n = {n}", ha="center", fontsize=7, color=SURFACE)
ax.set_xticks(range(len(vals)), [v[0].replace(" / ", " /\n") for v in vals], fontsize=8)
ax.set_ylabel("Held-out accuracy (%)")
ax.set_ylim(0, 90)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.tick_params(axis="x", length=0)
save(fig, "fig_capability_profile")

# ---------------------------------------------------------------- 7. visible-bleaching operating profile (held-out)


def profile(d, keys):
    keys = [k for k in keys if k in d and E.in_visible_bleaching_task(d[k]["question"], d[k]["question_type"])]
    tp = fn = tn = fp = nv = npos = nneg = 0
    for k in keys:
        r = d[k]
        g = E.label(r["question"], r["question_type"], r["answer_gt"])
        p = E.prediction(r["question"], r["question_type"], r.get("answer_pred"))
        if g not in ("visible_bleaching", "no_visible_bleaching"):
            continue
        if g == "visible_bleaching":
            npos += 1
        else:
            nneg += 1
        if p not in ("visible_bleaching", "no_visible_bleaching"):
            nv += 1
            continue
        if g == "visible_bleaching":
            tp += p == g; fn += p != g
        else:
            tn += p == g; fp += p != g
    ans = tp + fn + tn + fp
    # no-verdict answers count as not correct for sensitivity and specificity (as in the manuscript table)
    return {"Specificity": tn / nneg, "Answered-only\naccuracy": (tp + tn) / ans,
            "Precision\n(visible)": tp / max(1, tp + fp), "Sensitivity": tp / npos, "Coverage": ans / (ans + nv)}


sysb = [("CoralAgent-TRACE", A3, BLUE), ("Strict GPT-5", B0b, ORANGE), ("GPT-4.1", B1, AQUA), ("Grok-4.3", B2, YELLOW)]
profs = [(n, profile(d, H), c) for n, d, c in sysb]
mets = list(profs[0][1])
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=220)
w = 0.2
for j, (n, p, c) in enumerate(profs):
    xs_ = [i + (j - 1.5) * w for i in range(len(mets))]
    ax.bar(xs_, [p[m] for m in mets], width=w - 0.025, color=c, label=n, zorder=2)
    if j == 0:
        for x, m in zip(xs_, mets):
            ax.text(x, p[m] + 0.02, f"{p[m]:.2f}", ha="center", fontsize=6.5, fontweight="bold")
ax.set_xticks(range(len(mets)), mets, fontsize=8)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Rate")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
save(fig, "fig_bleaching_profile")
print("bleaching:", [(n, {k: round(v, 3) for k, v in p.items()}) for n, p, _ in profs])
print("paper figures written to", OUT)
