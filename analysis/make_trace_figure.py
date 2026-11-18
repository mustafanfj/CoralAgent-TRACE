"""Qualitative decision traces (replaces submitted Figure 9). Cases from analysis/trace_cases.json (seed 2026),
images are the CoralVQA bank images (set TRACE_IMAGES to their folder; files named <image_id>.jpg)."""
import json
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.environ.get("TRACE_IMAGES", os.path.join(HERE, "..", "reports", "trace_images"))
OUT = os.path.join(HERE, "figures")
BLUE, ORANGE, INK, MUTED, GREEN = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#1a7f4b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

cases = json.load(open(os.path.join(HERE, "trace_cases.json")))["figure_cases"]


def q(s):
    return re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", s).strip()


def cap(s):
    s = str(s if s not in (None, "") else "-")
    return s[:1].upper() + s[1:]


FW, FH = 7.4, 9.8
fig = plt.figure(figsize=(FW, FH), dpi=220)
PW, ROW = 0.47, 0.5                       # panel width (fraction of figure), row height
for n, c in enumerate(cases):
    x0p = 0.02 + (n % 2) * 0.51
    top = 1.0 - (n // 2) * ROW
    title = textwrap.fill(f"({c['panel']}) {c['title']}", 52)
    fig.text(x0p, top - 0.008, title, fontsize=8.2, fontweight="bold", va="top", color=INK)
    ax = fig.add_axes([x0p, top - 0.285, PW, 0.245])
    im = Image.open(os.path.join(IMG, f"{c['image_id']}.jpg"))
    w, h = im.size
    ax.imshow(im)
    ax.set_anchor("W")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if c.get("uses_crop") and c.get("bbox_norm") and c["bbox_norm"] != [0.0, 0.0, 1.0, 1.0]:
        bx0, by0, bx1, by1 = c["bbox_norm"]
        ax.add_patch(Rectangle((bx0 * w, by0 * h), (bx1 - bx0) * w, (by1 - by0) * h, fill=False, ec="#ffd400", lw=2.4))
        ax.text(bx0 * w + 0.015 * w, by1 * h - 0.03 * h, f"SCOPE crop: {c['region'].replace('_', '-')}", color=INK,
                fontsize=6.8, va="bottom", bbox=dict(fc="#ffd400", ec="none", pad=1.2))
    ver = cap(c.get("verifier_level"))
    if c.get("verifier_suggested") and str(c["verifier_suggested"]).lower() != str(c["final"]).lower():
        ver += f"; suggested '{c['verifier_suggested']}', repair blocked by the protection rule"
    focused = "Focused VQA (crop)" if c.get("uses_crop") and c["bbox_norm"] != [0.0, 0.0, 1.0, 1.0] else "Focused VQA"
    lines = [("Question", q(c["question"])),
             ("Tools", " > ".join(t.replace("_", " ") for t in (c.get("tools") or []))),
             ("Prior (strict GPT-5)", cap(c["prior"])),
             (focused, cap(c["focused_vqa"])),
             ("GUARD", ver),
             ("Final answer", cap(c["final"])),
             ("Reference", cap(c["reference"]))]
    y = top - 0.298
    for k, v in lines:
        wrapped = textwrap.wrap(v, 40) or ["-"]
        col = GREEN if k == "Final answer" else INK
        if k.startswith("Prior") and c["panel"] in ("a", "c"):
            col = ORANGE
        fig.text(x0p, y, k, fontsize=6.9, color=MUTED, va="top")
        fig.text(x0p + 0.155, y, "\n".join(wrapped), fontsize=6.9, color=col, va="top", linespacing=1.25,
                 fontweight="bold" if k == "Final answer" else "normal")
        y -= 0.0165 * len(wrapped) + 0.006
fig.savefig(os.path.join(OUT, "fig_qualitative_traces.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_qualitative_traces.pdf"), bbox_inches="tight")
print("written")
