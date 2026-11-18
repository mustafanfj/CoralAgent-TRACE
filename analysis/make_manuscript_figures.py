"""All result figures of the revised manuscript, in one consistent print style.

Every figure is drawn at its printed size: 7.0 in wide (the Frontiers text width, 504 pt), included at
width=\\linewidth, so fonts print at the sizes set here (smallest 8.5 pt). Numbers are recomputed from
results/v13; judge rates come from analysis/E8_judge.md and E8b_agreement.md, costs from H100_controls.md.

Outputs (analysis/figures/, png + pdf), in manuscript order:
  m02_paired_differences, m03_scale_consistency, m04_capability_profile, m05_bleaching_health,
  m06_accuracy_cost, m07_question_type_heatmap, m08_error_exposure, m09_selective_accuracy,
  m10_bleaching_profile, m11_decision_traces
"""
import csv
import glob
import gzip
import json
import os
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")
OUT = os.path.join(HERE, "figures")
IMG = os.environ.get("TRACE_IMAGES", os.path.join(HERE, "..", "reports", "trace_images"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
W = 7.0                                   # printed width in inches
BLUE, BLUE_LIGHT, GREY, GREY_DARK = "#2a78d6", "#b5d0f0", "#b9b7b0", "#8a8882"
INK, MUTED, GRID, SURFACE = "#111111", "#55534e", "#e6e5e1", "#ffffff"
GOOD, BAD, WARN, NEUTRAL = ("#e3f2e1", "#1e7b3a"), ("#fbe9e7", "#b8322a"), ("#fbf0d9", "#9a6700"), ("#efeeeb", "#6b6963")
INFO = ("#e6f0fb", "#1f5fa8")
CROP = "#f5c400"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.edgecolor": GREY_DARK, "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": INK, "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 8, "savefig.dpi": 300,
    "figure.dpi": 150, "pdf.fonttype": 42, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04})


def save(fig, name, tight=False):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


_sc = {}


def sc(r):
    k = (r["sample_id"], r.get("answer_pred"))
    if k not in _sc:
        _sc[k] = float(V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "",
                                              r["question_type"]) >= 0.5)
    return _sc[k]


def acc_ci(d, keys, n_boot=4000, seed=2026):
    by = {}
    for k in keys:
        by.setdefault(d[k]["image_id"], []).append(sc(d[k]))
    s = np.array([sum(v) for v in by.values()])
    n = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return 100 * s.sum() / n.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


A3 = load("A3v13_trace")
B0b, B1, B2 = load("B0b_strict_v13"), load("B1_gpt41_v13"), load("B2_grok_v13")
E5, E6, E7 = load("E5_single_call_same_info"), load("E6_self_consistency"), load("E7a_generic_gpt5_agent")
H = sorted(k for k, r in A3.items() if r["split"] == "H")
H100 = sorted(k for k in E6 if k in A3 and A3[k]["split"] == "H")
TYPES = [("bleaching_health", "Bleaching/health"), ("color_visual", "Colour"), ("genus_taxonomy", "Genus"),
         ("morphology_visual", "Morphology"), ("presence_count", "Presence/count"), ("other", "General")]

# ================================================================= Fig 2: paired differences
comparisons = [
    ("Zero-shot models, 2,400 held-out questions", [("Strict GPT-5 (primary test)", B0b, H), ("GPT-4.1", B1, H),
                                                    ("Grok-4.3", B2, H)]),
    ("GPT-5 controls, 600 matched held-out questions", [("Generic GPT-5 agent, same tools", E7, H100),
                                                        ("Single GPT-5 call, same information", E5, H100),
                                                        ("GPT-5 self-consistency (k = 3)", E6, H100),
                                                        ("Strict GPT-5", B0b, H100)])]
rows = []
for g, items in comparisons:
    for lab, d, keys in items:
        ks = [k for k in keys if k in d]
        bs = E.paired_bootstrap([(A3[k]["image_id"], sc(A3[k]), sc(d[k])) for k in ks])
        rows.append((g, lab, 100 * bs["diff"], 100 * bs["ci_low"], 100 * bs["ci_high"]))
fig, ax = plt.subplots(figsize=(W, 3.7))
y, ticks, labels, last = 0.0, [], [], None
ax.axvspan(0, 9, color=BLUE, alpha=0.05, lw=0, zorder=0)
for g, lab, d, lo, hi in rows:
    if g != last:
        y += 0.35 if last is not None else 0
        ax.text(-5.9, y, g, fontsize=10, fontweight="semibold", color=INK, va="center")
        y += 0.85
        last = g
    sig = lo > 0
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2.6, solid_capstyle="round", zorder=2)
    ax.plot([d], [y], "o", ms=8.5, color=BLUE if sig else SURFACE, mec=BLUE, mew=2, zorder=3)
    ax.text(hi + 0.25, y, f"{d:+.1f}", va="center", fontsize=10, color=INK, fontweight="semibold" if sig else "normal")
    ticks.append(y); labels.append(lab)
    y += 0.8
ax.axvline(0, color=GREY_DARK, lw=1)
ax.text(8.9, -0.35, "favours CoralAgent-TRACE  →", ha="right", fontsize=9, color=BLUE, va="center")
ax.set_yticks(ticks, labels)
ax.set_ylim(y - 0.2, -0.7)
ax.set_xlim(-6, 9)
ax.set_xlabel("CoralAgent-TRACE minus comparator (percentage points, 95% CI)")
xgrid(ax)
ax.spines["left"].set_visible(False)
ax.plot([], [], "o", color=BLUE, ms=8, label="CI excludes zero")
ax.plot([], [], "o", color=SURFACE, mec=BLUE, mew=2, ms=8, label="CI includes zero")
ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
save(fig, "m02_paired_differences")

# ================================================================= Fig 3: scale consistency
bank = (glob.glob(os.path.join(HERE, "..", "data_release", "*.csv"))
        or glob.glob(os.path.join(HERE, "..", "..", "coralagent-trace", "data_release", "*.csv")))
idx = {r["sample_id"]: int(r["active_image_index"]) for r in csv.DictReader(open(bank[0], encoding="utf-8"))}
by_img = {}
for k in H:
    if k in B0b:
        by_img.setdefault(idx[k], []).append(sc(A3[k]) - sc(B0b[k]))
order = sorted(by_img)
ds = np.array([sum(by_img[i]) for i in order]); cn = np.array([len(by_img[i]) for i in order])
rng = np.random.default_rng(2026)
xs, ys, los, his = [], [], [], []
for n in range(25, len(order) + 1, 25):
    bi = rng.integers(0, n, size=(4000, n))
    b = ds[:n][bi].sum(1) / cn[:n][bi].sum(1)
    xs.append(n); ys.append(100 * ds[:n].sum() / cn[:n].sum())
    los.append(100 * np.percentile(b, 2.5)); his.append(100 * np.percentile(b, 97.5))
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.fill_between(xs, los, his, color=BLUE_LIGHT, alpha=0.7, lw=0, zorder=1, label="95% image-cluster CI")
ax.plot(xs, ys, color=BLUE, lw=2.6, zorder=3, label="Cumulative accuracy difference")
ax.plot(xs[-1], ys[-1], "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=4)
ax.annotate(f"All 400 held-out images: {ys[-1]:+.1f} pp\n95% CI {los[-1]:+.1f} to {his[-1]:+.1f}, p = 0.011",
            (xs[-1], ys[-1]), xytext=(-40, -62), textcoords="offset points", ha="right", fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=GREY_DARK, lw=0.8))
ax.axhline(0, color=GREY_DARK, lw=1)
ax.set_xlim(15, 410)
ax.set_xticks([25, 100, 200, 300, 400])
ax.set_xlabel("Held-out images evaluated, in bank order (6 questions each)")
ax.set_ylabel("TRACE minus strict\nGPT-5 (pp)")
ygrid(ax)
ax.legend(frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
save(fig, "m03_scale_consistency")

# ================================================================= Fig 4: capability profile
vals = []
for t, lab in TYPES:
    ks = [k for k in H if A3[k]["question_type"] == t]
    a, lo, hi = acc_ci(A3, ks)
    vals.append((lab, a, lo, hi, len(ks)))
vals.sort(key=lambda v: -v[1])
fig, ax = plt.subplots(figsize=(W, 2.8))
for i, (lab, a, lo, hi, n) in enumerate(vals):
    ax.barh(i, a, height=0.62, color=BLUE, zorder=2)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.2, zorder=3)
    ax.text(hi + 1.2, i, f"{a / 100:.3f}", va="center", fontsize=10, fontweight="semibold")
ax.set_yticks(range(len(vals)), [f"{v[0]}  (n = {v[4]})" for v in vals])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("CoralAgent-TRACE accuracy on the 2,400 held-out questions (%), 95% CI")
xgrid(ax)
save(fig, "m04_capability_profile")

# ================================================================= Fig 5: bleaching / health
BH = [k for k in H if A3[k]["question_type"] == "bleaching_health"]
BH100 = [k for k in H100 if A3[k]["question_type"] == "bleaching_health"]
groups = [(f"Zero-shot models, {len(BH)} held-out questions", BH,
           [("CoralAgent-TRACE", A3), ("Strict GPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]),
          (f"GPT-5 controls, {len(BH100)} matched held-out questions", BH100,
           [("CoralAgent-TRACE", A3), ("Generic GPT-5 agent, same tools", E7), ("Single GPT-5 call, same information", E5),
            ("GPT-5 self-consistency (k = 3)", E6), ("Strict GPT-5", B0b)])]
fig, ax = plt.subplots(figsize=(W, 4.1))
y, ticks, labels = 0.0, [], []
for gi, (gname, keys, items) in enumerate(groups):
    y += 0.4 if gi else 0
    ax.text(0.5, y, gname, fontsize=10, fontweight="semibold", va="center", ha="left", color=INK)
    y += 0.85
    for i_, (lab, d) in enumerate(items):
        a, lo, hi = acc_ci(d, [k for k in keys if k in d])
        main = i_ == 0
        ax.barh(y, a, height=0.66, color=BLUE if main else GREY, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=3)
        ax.text(hi + 1.0, y, f"{a / 100:.3f}", va="center", fontsize=10.5 if main else 10,
                fontweight="bold" if main else "normal")
        ticks.append(y); labels.append(lab)
        y += 0.82
ax.set_yticks(ticks, labels)
for t_, lab in zip(ax.get_yticklabels(), labels):
    if lab == "CoralAgent-TRACE":
        t_.set_fontweight("bold")
ax.set_ylim(y - 0.3, -0.6)
ax.set_xlim(0, 100)
ax.set_xlabel("Accuracy on bleaching/health questions (%), 95% image-cluster CI")
xgrid(ax)
ax.spines["left"].set_visible(False)
save(fig, "m05_bleaching_health")

# ================================================================= Fig 6: accuracy vs cost
cost = [("CoralAgent-TRACE", A3, 0.0095, (12, 14), "left"),
        ("Generic GPT-5 agent\n(same tools and verifier)", E7, 0.0109, (12, -16), "left"),
        ("Single GPT-5 call,\nsame information", E5, 0.0020, (0, -52), "center"),
        ("GPT-5 self-consistency (k = 3)", E6, 0.0034, (12, 10), "left"),
        ("Strict GPT-5", B0b, 0.0011, (0, 46), "center")]
fig, ax = plt.subplots(figsize=(W, 3.6))
pts = {}
for name, d, c, off, ha in cost:
    a, lo, hi = acc_ci(d, [k for k in H100 if k in d])
    main = name == "CoralAgent-TRACE"
    pts[name] = (c, a)
    ax.plot([c, c], [lo, hi], color=BLUE if main else GREY, lw=1.6, zorder=2)
    ax.plot(c, a, "o", ms=11 if main else 9, color=BLUE if main else GREY_DARK, mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f"{name}\n{a / 100:.3f}", (c, a), textcoords="offset points", xytext=off, ha=ha, va="center",
                fontsize=10 if main else 9.5, fontweight="bold" if main else "normal", zorder=6,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=1.5))
(cg, ag), (ct, at) = pts["Generic GPT-5 agent\n(same tools and verifier)"], pts["CoralAgent-TRACE"]
ax.annotate("", xy=(ct + 0.00015, at - 0.35), xytext=(cg - 0.00005, ag + 0.45),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, mutation_scale=14, connectionstyle="arc3,rad=0.3"))
ax.text(0.0091, 50.6, "+4.2 pp accuracy\n\u221213% cost", color=BLUE, fontsize=9.5, fontweight="semibold",
        va="center", ha="right")
ax.set_xlim(0, 0.0145)
ax.set_ylim(42, 61)
ax.set_yticks([44, 48, 52, 56, 60])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.3f}"))
ax.set_xlabel("Cost per question (USD, batch prices)")
ax.set_ylabel("Accuracy (%), 95% CI")
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
save(fig, "m06_accuracy_cost")

# ================================================================= Fig 7: question-type heatmap
methods = [("CoralAgent-TRACE", A3), ("Generic GPT-5 agent", E7), ("Single GPT-5 call", E5),
           ("GPT-5 self-consistency", E6), ("Strict GPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]
M = np.zeros((len(methods), len(TYPES))); ns = []
for j, (t, _) in enumerate(TYPES):
    ks = [k for k in H100 if A3[k]["question_type"] == t]; ns.append(len(ks))
    for i, (_, d) in enumerate(methods):
        M[i, j] = 100 * np.mean([sc(d[k]) for k in ks])
fig, ax = plt.subplots(figsize=(W, 3.6))
im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9cc3ef", "#1f5fa8"]),
               vmin=15, vmax=85, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        best = M[i, j] >= M[:, j].max() - 1e-9
        ax.text(j, i, f"{M[i, j] / 100:.2f}", ha="center", va="center", fontsize=10.5,
                color=SURFACE if M[i, j] > 58 else INK, fontweight="bold" if best else "normal")
ax.add_patch(Rectangle((-0.5, -0.5), len(TYPES), 1, fill=False, ec=INK, lw=1.6, zorder=5))
ax.set_xticks(range(len(TYPES)), [f"{lab.replace('/', '/\n') if '/' in lab else lab + chr(10)}\n(n = {n})" for (_, lab), n in zip(TYPES, ns)], fontsize=9.5)
ax.xaxis.tick_top()
ax.set_yticks(range(len(methods)), [m[0] for m in methods])
ax.get_yticklabels()[0].set_fontweight("bold")
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.axhline(3.5 + 0.0, color=SURFACE, lw=3)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.set_label("Accuracy (%)"); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)
save(fig, "m07_question_type_heatmap")

# ================================================================= Fig 8: error exposure
panels = [("(a) Errors flagged\nin the trace\n(higher is better)", "",
           [("Grok-4.3 judge", 0.564, 0.255, "+0.31"), ("GPT-4.1 judge", 0.341, 0.104, "+0.24")]),
          ("(b) Errors that look\nconvincing\n(lower is better)", "",
           [("Grok-4.3 judge", 0.337, 0.570, "−0.23"), ("GPT-4.1 judge", 0.568, 0.809, "−0.24")]),
          ("(c) Evidence predicts\ncorrectness\n(AUROC, higher is better)", "",
           [("Grok-4.3 judge", 0.674, 0.590, "+0.08"), ("GPT-4.1 judge", 0.639, 0.536, "+0.10")])]
fig, axes = plt.subplots(1, 3, figsize=(W, 3.5), sharey=True, gridspec_kw={"wspace": 0.05})
for ax, (title, note, rws) in zip(axes, panels):
    w = 0.38
    for i, (judge, t, g, diff) in enumerate(rws):
        ax.bar(i - w / 2 - 0.01, t, width=w, color=BLUE, zorder=2)
        ax.bar(i + w / 2 + 0.01, g, width=w, color=GREY, zorder=2)
        ax.text(i - w / 2 - 0.01, t + 0.025, f"{t:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w / 2 + 0.01, g + 0.025, f"{g:.2f}", ha="center", fontsize=10)
        ax.text(i, 0.99, f"Δ {diff}", ha="center", fontsize=9.5, color=BLUE, fontweight="semibold")
    ax.set_xticks(range(2), [r[0].replace(" judge", "\njudge") for r in rws])
    ax.set_title(title, fontsize=9.5, linespacing=1.3)
    ax.set_ylim(0, 1.06)
    ygrid(ax)
axes[0].set_ylabel("Share of incorrect answers / AUROC")
for ax in axes[1:]:
    ax.tick_params(axis="y", left=False)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GREY)],
           labels=["CoralAgent-TRACE", "Generic GPT-5 agent (same tools and verifier)"], frameon=False, ncol=2,
           loc="outside lower center")
save(fig, "m08_error_exposure")

# ================================================================= Fig 9: selective accuracy
ja, je = load("E8_judge_A3v13"), load("E8_judge_E7a")
keys = sorted(k for k in ja if k in je)


def curve(j, s):
    rk = sorted(keys, key=lambda k: (bool(j[k]["rating"].get("contradicts_answer")), -float(j[k]["rating"]["overall"]), k))
    xs_, ys_ = [], []
    for i in range(60, len(rk) + 1, 6):
        xs_.append(100 * i / len(rk)); ys_.append(100 * sum(sc(s[k]) for k in rk[:i]) / i)
    return np.array(xs_), np.array(ys_)


xt, yt = curve(ja, A3)
xg, yg = curve(je, E7)
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.fill_between(xt, yg, yt, where=yt >= yg, color=BLUE, alpha=0.08, lw=0, zorder=1)
ax.plot(xg, yg, color=GREY_DARK, lw=2.4, zorder=2)
ax.plot(xt, yt, color=BLUE, lw=2.8, zorder=3)
ax.text(xt[-1] + 1.2, yt[-1], "CoralAgent-TRACE", color=INK, fontsize=10, fontweight="bold", va="center")
ax.text(xg[-1] + 1.2, yg[-1], "Generic GPT-5 agent", color=INK, fontsize=10, va="center")
i60 = int(np.argmin(abs(xt - 60)))
ax.plot([60, 60], [yg[i60], yt[i60]], color=INK, lw=1, ls=(0, (2, 2)))
ax.plot(60, yt[i60], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4)
ax.plot(60, yg[i60], "o", ms=7, color=GREY_DARK, mec=SURFACE, mew=2, zorder=4)
ax.annotate(f"Accept 60%: {yt[i60] / 100:.3f} vs {yg[i60] / 100:.3f}", (60, yt[i60]), xytext=(8, 14),
            textcoords="offset points", fontsize=9.5, fontweight="semibold", color=BLUE)
ax.set_xlim(10, 128)
ax.set_xticks([20, 40, 60, 80, 100])
ax.set_xlabel("Answers accepted (%); the rest, with the least trusted evidence, are referred to an expert")
ax.set_ylabel("Accuracy of accepted\nanswers (%)")
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
save(fig, "m09_selective_accuracy")

# ================================================================= Fig 10: bleaching operating profile
tp = fn = tn = fp = nv = 0
for k in H:
    r = A3[k]
    if not E.in_visible_bleaching_task(r["question"], r["question_type"]):
        continue
    g = E.label(r["question"], r["question_type"], r["answer_gt"])
    p_ = E.prediction(r["question"], r["question_type"], r.get("answer_pred"))
    if g not in ("visible_bleaching", "no_visible_bleaching"):
        continue
    if p_ not in ("visible_bleaching", "no_visible_bleaching"):
        nv += 1
    elif g == "visible_bleaching":
        tp += p_ == g; fn += p_ != g
    else:
        tn += p_ == g; fp += p_ != g
tot = tp + fn + tn + fp + nv
prof = [("Binary coverage", (tot - nv) / tot), ("Specificity (unbleached cleared)", tn / (tn + fp)),
        ("Accuracy when answered", (tp + tn) / (tot - nv)), ("Strict accuracy", (tp + tn) / tot),
        ("Strict bleached recall", tp / (tp + fn)), ("False-bleaching-alarm rate", fp / (tn + fp))]
fig, ax = plt.subplots(figsize=(W, 2.8))
for i, (lab, v) in enumerate(prof):
    ax.barh(i, v, height=0.62, color=BLUE, zorder=2)
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=10, fontweight="semibold")
ax.set_yticks(range(len(prof)), [p[0] for p in prof])
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel(f"Rate on the {tot} held-out visible-bleaching questions")
xgrid(ax)
save(fig, "m10_bleaching_profile")
print("charts done; bleaching:", tp, fn, tn, fp, nv)

# ================================================================= Fig 11: decision traces
cases = json.load(open(os.path.join(HERE, "trace_cases.json")))["figure_cases"]


def qtext(s):
    return re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", s).strip()


def cap(s):
    s = str(s if s not in (None, "") else "-")
    return s[:1].upper() + s[1:]


def same(a, b):
    return str(a).strip().lower() == str(b).strip().lower()


def square(im, bb):
    w, h = im.size
    s = min(w, h)
    x0, y0 = 0, (h - s) // 2
    im = im.crop((x0, y0, x0 + s, y0 + s))
    if bb:
        bb = [max(0, (bb[0] * w - x0) / s), max(0, (bb[1] * h - y0) / s), min(1, (bb[2] * w - x0) / s),
              min(1, (bb[3] * h - y0) / s)]
    return im, bb


TOOLN = {"spatial_focus": "SCOPE", "baseline_prior": "zero-shot prior", "focused_vqa": "focused VQA",
         "retrieval": "ecological retrieval", "genus_identification": "genus identification"}
RH, IS, GAPR, LEG = 1.98, 1.78, 0.1, 0.36
FH = 4 * RH + 3 * GAPR + LEG
fig = plt.figure(figsize=(W, FH), layout="none")
base = fig.add_axes([0, 0, 1, 1]); base.set_xlim(0, W); base.set_ylim(0, FH); base.axis("off")


def fax(x, y, w, h):
    return fig.add_axes([x / W, y / FH, w / W, h / FH])


def chip(x, y, w, h, style, label, value, mark="", strong=False):
    fill, edge = style
    base.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fill, ec=edge,
                                  lw=2.0 if strong else 1.0, zorder=3))
    base.text(x + w / 2, y + h - 0.1, label, ha="center", va="top", fontsize=8.5, color=MUTED, zorder=4)
    txt = f"{value} {mark}".strip()
    fs = 11 if len(txt) <= 11 else (9.8 if len(txt) <= 15 else 8.8)
    base.text(x + w / 2, y + 0.22, txt, ha="center", va="center", fontsize=fs, color=edge, fontweight="bold",
              zorder=4, linespacing=1.0)


for n, c in enumerate(cases):
    y0 = FH - (n + 1) * RH - n * GAPR
    base.add_patch(FancyBboxPatch((0.01, y0), W - 0.02, RH, boxstyle="round,pad=0,rounding_size=0.08", fc="#fafaf8",
                                  ec="#dcdad5", lw=0.9, zorder=0))
    crop_used = c.get("uses_crop") and c.get("bbox_norm") and c["bbox_norm"] != [0.0, 0.0, 1.0, 1.0]
    im, bb = square(Image.open(os.path.join(IMG, f"{c['image_id']}.jpg")).convert("RGB"),
                    c["bbox_norm"] if crop_used else None)
    ix, iy = 0.1, y0 + (RH - IS) / 2
    ax = fax(ix, iy, IS, IS); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    S = im.size[0]
    if crop_used:
        ax.add_patch(Rectangle((bb[0] * S, bb[1] * S), (bb[2] - bb[0]) * S, (bb[3] - bb[1]) * S, fill=False,
                               ec=CROP, lw=2.6))
    rx = ix + IS + 0.2
    rw = W - rx - 0.12
    top = y0 + RH - 0.13
    base.text(rx, top, f"({c['panel']})", fontsize=11, fontweight="bold", color=BLUE, va="top")
    base.text(rx + 0.32, top, c["title"], fontsize=11, fontweight="bold", color=INK, va="top")
    qwidth = 74
    qlines = textwrap.wrap("“" + qtext(c["question"]) + "”", qwidth)
    base.text(rx, top - 0.3, "\n".join(qlines), fontsize=10, style="italic", color=INK, va="top", linespacing=1.2)
    ly = top - 0.3 - 0.2 * len(qlines) - 0.06
    tools = " → ".join(TOOLN.get(t, t) for t in (c.get("tools") or []))
    base.text(rx, ly, "Reference answer: ", fontsize=9.5, color=MUTED, va="top")
    base.text(rx + 1.32, ly, cap(c["reference"]), fontsize=10, color=INK, va="top", fontweight="bold")
    base.text(rx, ly - 0.22, f"Tools: {tools}", fontsize=9, color=MUTED, va="top")
    if crop_used:                                   # zoomed SCOPE crop, inset at the photo's bottom-right
        zs = 0.92
        zx, zy = ix + IS - zs - 0.05, iy + 0.05
        zax = fax(zx, zy, zs, zs)
        x0p, y0p, x1p, y1p = [int(v * S) for v in bb]
        zax.imshow(im.crop((x0p, y0p, x1p, y1p))); zax.set_xticks([]); zax.set_yticks([])
        for s_ in zax.spines.values():
            s_.set_color(CROP); s_.set_linewidth(2.6)
        for (px, py), (qx, qy) in [((bb[2] * S, bb[1] * S), (1, 1)), ((bb[0] * S, bb[3] * S), (0, 1))]:
            con = ConnectionPatch(xyA=(px, py), coordsA=ax.transData, xyB=(qx, qy), coordsB=zax.transAxes,
                                  color=CROP, lw=1.2)
            con.set_zorder(10); fig.add_artist(con)
        zax.text(0.04, 0.04, "SCOPE crop", transform=zax.transAxes, fontsize=8, color=INK, va="bottom",
                 bbox=dict(fc=CROP, ec="none", pad=1.5))
    # decision chain
    fv = str(c["focused_vqa"] or "")
    fv_style, fv_mark = ((NEUTRAL, "?") if fv.lower() in ("", "unclear", "none")
                         else ((GOOD, "✓") if same(fv, c["reference"]) else (BAD, "✗")))
    blocked = bool(c.get("verifier_suggested")) and not same(c["verifier_suggested"], c["final"])
    chain = [("Zero-shot prior", *(((GOOD, "✓") if same(c["prior"], c["reference"]) else (BAD, "✗"))),
              cap(c["prior"])),
             ("Focused VQA", fv_style, fv_mark, cap(fv) if fv else "-"),
             ("GUARD check", WARN if blocked else INFO, "",
              "Rejected, repair\nblocked" if blocked else cap(c.get("verifier_level"))),
             ("Final answer", *(((GOOD, "✓") if same(c["final"], c["reference"]) else (BAD, "✗"))),
              cap(c["final"]))]
    cw, ch, cgap = 1.03, 0.66, 0.16
    cy = y0 + 0.13
    for k, (lab, st, mk, val) in enumerate(chain):
        cx = rx + k * (cw + cgap)
        chip(cx, cy, cw, ch, st, lab, val, mk, strong=(k == 3))
        if k < 3:
            base.add_patch(FancyArrowPatch((cx + cw + 0.03, cy + ch / 2), (cx + cw + cgap - 0.03, cy + ch / 2),
                                           arrowstyle="-|>", mutation_scale=11, color=GREY_DARK, lw=1.2, zorder=2))
    if blocked:
        base.text(rx + 2.4, ly, f"Verifier suggested: {c['verifier_suggested']}", fontsize=9.5, color=WARN[1],
                  va="top", fontweight="semibold")

# legend
items = [(GOOD, "\u2713", "correct"), (BAD, "\u2717", "incorrect"), (NEUTRAL, "?", "no definite answer"),
         (INFO, "", "GUARD verdict"), (WARN, "", "repair blocked")]
lx = 0.3
for st, mk, lab in items:
    base.add_patch(FancyBboxPatch((lx, 0.09), 0.24, 0.19, boxstyle="round,pad=0,rounding_size=0.04", fc=st[0],
                                  ec=st[1], lw=1.0))
    base.text(lx + 0.12, 0.185, mk, ha="center", va="center", fontsize=9, color=st[1], fontweight="bold")
    t = base.text(lx + 0.31, 0.185, lab, va="center", fontsize=9, color=INK)
    lx += 0.31 + 0.062 * len(lab) + 0.32
save(fig, "m11_decision_traces", tight=False)
print("all figures written to", OUT)
