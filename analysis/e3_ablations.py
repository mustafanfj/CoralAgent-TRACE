"""E3 ablations on D (reviewer comment C5): each variant vs the full system A3v13 on the same D questions.

All variants are built on A3v13 (v13 fix 4). A5 (answer before GUARD) is derived from the E2 run.
Statistics: paired image-cluster bootstrap + exact McNemar, Holm across the ablation family.
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import v12_core as V  # noqa: E402
from coralagent import evaluation as E  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")
VARIANTS = {
    "A0 SCOPE only (no prior, no tools beyond the crop)": "A0v13_scope_only",
    "A1 prior + SCOPE": "A1v13_prior_scope",
    "A2 fixed: all tools on every question": "A2v13_fixed_all_tools",
    "A4 no retrieval": "A4v13_no_retrieval",
    "A5 no GUARD (answer before verification)": "A5v13_no_guard",
    "A6 no baseline prior": "A6v13_no_prior",
    "A3 without SCOPE crops": "A3v13_no_scope",
}


def load(m):
    p = os.path.join(R, f"{m}.jsonl.gz")
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(p))} if os.path.exists(p) else {}


def sc(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"])


A3 = load("A3v13_trace")
D = {k for k, r in A3.items() if r["split"] == "D"}
data = {name: load(m) for name, m in VARIANTS.items()}
missing = [n for n, d in data.items() if not d]
if missing:
    sys.exit(f"Not exported yet: {missing}")
common = sorted(D.intersection(*[set(d) for d in data.values()]))
print(f"D questions present for A3v13 and every variant: {len(common)} "
      f"(images: {len({A3[k]['image_id'] for k in common})})\n")

print(f"| Method | Accuracy (n={len(common)}) |")
print("|---|---|")
print(f"| A3v13 full system | {sum(sc(A3[k]) for k in common) / len(common):.3f} |")
for name, d in data.items():
    print(f"| {name} | {sum(sc(d[k]) for k in common) / len(common):.3f} |")

print("\n| Variant minus A3v13 | Difference | 95% CI | variant better/worse | p | Holm p |")
print("|---|---|---|---|---|---|")
res, pv = {}, {}
for name, d in data.items():
    rows = [(A3[k]["image_id"], sc(d[k]), sc(A3[k])) for k in common]
    bs = E.paired_bootstrap(rows)
    b = sum(1 for _, x, y in rows if x >= 0.5 > y)
    c = sum(1 for _, x, y in rows if y >= 0.5 > x)
    pv[name] = E.mcnemar_exact(b, c)
    res[name] = (bs, b, c)
holm = E.holm(pv)
for name, (bs, b, c) in res.items():
    print(f"| {name} | {bs['diff']:+.3f} | {bs['ci_low']:+.3f} to {bs['ci_high']:+.3f} | {b}/{c} | "
          f"{pv[name]:.3g} | {holm[name]:.3g} |")

types = sorted({A3[k]["question_type"] for k in common})
print("\nBy question type (accuracy):")
print("| Method | " + " | ".join(types) + " |")
print("|---|" + "---|" * len(types))
for name, d in [("A3v13 full system", A3)] + list(data.items()):
    cells = []
    for t in types:
        ks = [k for k in common if A3[k]["question_type"] == t]
        cells.append(f"{sum(sc(d[k]) for k in ks) / len(ks):.3f} ({len(ks)})")
    print(f"| {name} | " + " | ".join(cells) + " |")
