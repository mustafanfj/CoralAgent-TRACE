"""Qualitative decision traces (revised Figure 11; replaces submitted Figure 9).

Cases come from analysis/trace_cases.json (random, seed 2026, within each category). Images are the CoralVQA bank
images; set TRACE_IMAGES to their folder (files named <image_id>.jpg). Defaults to reports/trace_images/, where
the lab-PC stage IMG exports them.
"""
import json
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.environ.get("TRACE_IMAGES", os.path.join(HERE, "..", "reports", "trace_images"))
OUT = os.path.join(HERE, "figures")

INK, MUTED, BAND, CARD, EDGE = "#14212b", "#5b6670", "#24507a", "#fbfbfa", "#d9dcdf"
STYLE = {"ok": ("#e5f4e4", "#1a7f4b", "✓"), "bad": ("#fbebeb", "#c0392b", "✗"),
         "none": ("#f0efec", "#8a8984", "?"), "guard": ("#e7f1fb", "#2a78d6", ""),
         "blocked": ("#f9eeda", "#b7791f", "")}
CROP = "#ffd400"
plt.rcParams.update({"font.family": "DejaVu Sans"})

cases = json.load(open(os.path.join(HERE, "trace_cases.json")))["figure_cases"]


def q(s):
    return re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", s).strip()


def cap(s):
    s = str(s if s not in (None, "") else "-")
    return s[:1].upper() + s[1:]


def same(a, b):
    return str(a).strip().lower() == str(b).strip().lower()


def square(im, bbox):
    """Centre-crop to a square; return the image and the bbox in the new normalised coordinates."""
    w, h = im.size
    s = min(w, h)
    x0, y0 = 0, (h - s) // 2                 # keep the left edge (scale bars sit bottom-left)
    im = im.crop((x0, y0, x0 + s, y0 + s))
    if bbox:
        bx0, by0, bx1, by1 = bbox
        bbox = [max(0, (bx0 * w - x0) / s), max(0, (by0 * h - y0) / s), min(1, (bx1 * w - x0) / s), min(1, (by1 * h - y0) / s)]
    return im, bbox


W, CW, GAP, M = 7.4, 3.6, 0.2, 0.0          # figure width, card width, gap between cards, outer margin (inches)
CH = 3.33                                    # card height
LEG = 0.42                                   # legend strip at the bottom
H = 2 * CH + GAP + LEG
fig = plt.figure(figsize=(W, H), dpi=250)
base = fig.add_axes([0, 0, 1, 1])
base.set_xlim(0, W); base.set_ylim(0, H); base.axis("off")


def fax(x, y, w, h):
    """Axes at an absolute position in inches (origin bottom-left)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def box(x, y, w, h, kind, label, value, bold_edge=False):
    fill, edge, mark = STYLE[kind]
    base.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=edge,
                                  lw=1.8 if bold_edge else 0.9, zorder=3))
    base.text(x + w / 2, y + h - 0.1, label, ha="center", va="top", fontsize=5.6, color=MUTED, zorder=4)
    val = "\n".join(textwrap.wrap(value, 12)[:2])
    txt = f"{val} {mark}".strip() if mark else val
    base.text(x + w / 2, y + 0.2, txt, ha="center", va="center", fontsize=7.4 if len(value) < 9 else 6.6,
              color=edge if kind != "guard" else INK, fontweight="bold", zorder=4, linespacing=1.05)


for n, c in enumerate(cases):
    cx = M + (n % 2) * (CW + GAP)
    cy = LEG + (1 - n // 2) * (CH + GAP)
    # card and title band
    base.add_patch(FancyBboxPatch((cx, cy), CW, CH, boxstyle="round,pad=0,rounding_size=0.08", fc=CARD, ec=EDGE,
                                  lw=0.8, zorder=0))
    base.add_patch(FancyBboxPatch((cx, cy + CH - 0.42), CW, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                  fc=BAND, ec=BAND, zorder=1))
    base.add_patch(Rectangle((cx, cy + CH - 0.42), CW, 0.12, fc=BAND, ec=BAND, zorder=1))
    base.text(cx + 0.12, cy + CH - 0.21, f"({c['panel']})", color="white", fontsize=8.5, fontweight="bold",
              va="center", zorder=2)
    base.text(cx + 0.42, cy + CH - 0.21, "\n".join(textwrap.wrap(c["title"], 44)), color="white", fontsize=7.3,
              va="center", zorder=2, linespacing=1.1)

    # image (square), with the SCOPE crop marked when a crop was used
    crop_used = c.get("uses_crop") and c.get("bbox_norm") and c["bbox_norm"] != [0.0, 0.0, 1.0, 1.0]
    im, bb = square(Image.open(os.path.join(IMG, f"{c['image_id']}.jpg")).convert("RGB"),
                    c["bbox_norm"] if crop_used else None)
    S = 1.95
    ix, iy = cx + 0.12, cy + CH - 0.52 - S
    ax = fax(ix, iy, S, S)
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    s = im.size[0]
    if crop_used:
        ax.add_patch(Rectangle((bb[0] * s, bb[1] * s), (bb[2] - bb[0]) * s, (bb[3] - bb[1]) * s, fill=False,
                               ec=CROP, lw=2.2))

    # right column: question, reference, and the SCOPE crop inset (panel c)
    rx, rw = ix + S + 0.13, CW - S - 0.37
    top = cy + CH - 0.58
    base.text(rx, top, "Question", fontsize=5.8, color=MUTED, va="top")
    qtxt = "\n".join(textwrap.wrap(q(c["question"]), 24))
    base.text(rx, top - 0.14, qtxt, fontsize=7.0, color=INK, va="top", style="italic", linespacing=1.15)
    nq = qtxt.count("\n") + 1
    ry = top - 0.2 - 0.135 * nq - 0.08
    base.text(rx, ry, "Reference answer", fontsize=5.8, color=MUTED, va="top")
    base.text(rx, ry - 0.14, cap(c["reference"]), fontsize=8.2, color=INK, va="top", fontweight="bold")
    if crop_used:
        zs = min(rw, ry - 0.42 - (iy + 0.02))
        zx, zy = rx, iy
        zax = fax(zx, zy, zs, zs)
        x0, y0, x1, y1 = [int(v * s) for v in bb]
        zax.imshow(im.crop((x0, y0, x1, y1)))
        zax.set_xticks([]); zax.set_yticks([])
        for sp in zax.spines.values():
            sp.set_color(CROP); sp.set_linewidth(2.2)
        base.text(zx, zy + zs + 0.04, f"SCOPE crop: {c['region'].replace('_', '-')}", fontsize=5.8, color=MUTED,
                  va="bottom")
        # connector from the crop box to the inset
        for (px, py), (qx, qy) in [((bb[2] * s, bb[1] * s), (0, 0)), ((bb[2] * s, bb[3] * s), (0, 1))]:
            con = ConnectionPatch(xyA=(px, py), coordsA=ax.transData, xyB=(qx, qy), coordsB=zax.transAxes,
                                  color=CROP, lw=1.1, alpha=0.95)
            con.set_zorder(10)
            fig.add_artist(con)
    else:
        names = {"spatial_focus": "SCOPE spatial focus", "baseline_prior": "Zero-shot prior", "focused_vqa": "Focused VQA",
                 "retrieval": "Ecological retrieval", "genus_identification": "Genus identification"}
        tools = [names.get(t, t.replace("_", " ")) for t in (c.get("tools") or [])]
        base.text(rx, ry - 0.42, "Tools called", fontsize=5.8, color=MUTED, va="top")
        base.text(rx, ry - 0.56, "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tools)), fontsize=6.2, color=INK,
                  va="top", linespacing=1.4)

    # decision chain: prior -> focused VQA -> GUARD -> final
    fv = str(c["focused_vqa"] or "")
    fv_kind = "none" if fv.lower() in ("", "unclear", "none") else ("ok" if same(fv, c["reference"]) else "bad")
    ver = str(c.get("verifier_level") or "")
    blocked = bool(c.get("verifier_suggested")) and not same(c["verifier_suggested"], c["final"])
    g_kind, g_val = ("blocked", "Unsupported,\nrepair blocked") if blocked else ("guard", cap(ver).replace("_", " "))
    steps = [("Zero-shot prior", "ok" if same(c["prior"], c["reference"]) else "bad", cap(c["prior"])),
             ("Focused VQA", fv_kind, cap(fv) if fv else "-"),
             ("GUARD check", g_kind, g_val),
             ("Final answer", "ok" if same(c["final"], c["reference"]) else "bad", cap(c["final"]))]
    bw, bh, by = 0.76, 0.6, cy + 0.12
    gap = (CW - 0.24 - 4 * bw) / 3
    for k, (lab, kind, val) in enumerate(steps):
        bx = cx + 0.12 + k * (bw + gap)
        if kind == "blocked":
            fill, edge, _ = STYLE[kind]
            base.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.06", fc=fill,
                                          ec=edge, lw=0.9, zorder=3))
            base.text(bx + bw / 2, by + bh - 0.1, lab, ha="center", va="top", fontsize=5.6, color=MUTED, zorder=4)
            base.text(bx + bw / 2, by + 0.22, val, ha="center", va="center", fontsize=6.0, color=edge,
                      fontweight="bold", zorder=4, linespacing=1.05)
        else:
            box(bx, by, bw, bh, kind, lab, val, bold_edge=(k == 3))
        if k < 3:
            base.add_patch(FancyArrowPatch((bx + bw + 0.02, by + bh / 2), (bx + bw + gap - 0.02, by + bh / 2),
                                           arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.9, zorder=2))
    if blocked:
        base.text(cx + 0.12 + 2 * (bw + gap) + bw / 2, by - 0.005, f"suggested '{c['verifier_suggested']}'",
                  ha="center", va="top", fontsize=5.2, color=STYLE["blocked"][1], style="italic")

# legend
items = [("ok", "matches the reference"), ("bad", "differs from the reference"), ("none", "no definite answer"),
         ("guard", "GUARD verdict"), ("blocked", "repair blocked by protection")]
lx = 0.15
for kind, lab in items:
    fill, edge, mark = STYLE[kind]
    base.add_patch(FancyBboxPatch((lx, 0.13), 0.2, 0.16, boxstyle="round,pad=0,rounding_size=0.03", fc=fill, ec=edge,
                                  lw=0.9))
    base.text(lx + 0.1, 0.21, mark, ha="center", va="center", fontsize=6.5, color=edge, fontweight="bold")
    t = base.text(lx + 0.26, 0.21, lab, va="center", fontsize=6.3, color=INK)
    lx += 0.26 + 0.047 * len(lab) + 0.18

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_qualitative_traces.{ext}"))
print("written")
