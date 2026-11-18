"""H100 controls (reviewer comments C1, C2, C6): CoralAgent-TRACE vs matched GPT-5 controls on the same 600 questions.

E5  single GPT-5 call given the same information (image, SCOPE crop, retrieved records, rules, genus list)
E6  strict GPT-5 self-consistency, k=3 majority vote (matched call budget)
E7a generic GPT-5 tool-using agent with the same toolbox and GUARD
B0b strict GPT-5 (E1), restricted to H100

Secondary comparisons: paired image-cluster bootstrap + exact McNemar, Holm-corrected as one family.
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


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


def sc(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"])


A3 = load("A3v13_trace")
A5 = load("A5v13_no_guard")
others = {
    "E5 single call, same information": load("E5_single_call_same_info"),
    "E6 strict GPT-5 self-consistency (k=3)": load("E6_self_consistency"),
    "E7a generic GPT-5 agent": load("E7a_generic_gpt5_agent"),
    "Strict GPT-5 (B0b)": load("B0b_strict_v13"),
}
# E6 ran on exactly the 600 pre-registered H100 questions (E5 and E7a also hold their 15 E0 pilot rows from D)
H100 = sorted(k for k in others["E6 strict GPT-5 self-consistency (k=3)"] if k in A3 and A3[k]["split"] == "H")
print(f"H100 questions: {len(H100)}\n")

print("| Method | n | Accuracy |")
print("|---|---|---|")
for name, d in [("CoralAgent-TRACE (A3v13)", A3), ("A3v13 before GUARD (A5)", A5)] + list(others.items()):
    ks = [k for k in H100 if k in d]
    print(f"| {name} | {len(ks)} | " + (f"{sum(sc(d[k]) for k in ks) / len(ks):.3f} |" if ks else "not run yet |"))

print("\n| A3v13 minus | n | Difference | 95% CI | McNemar b/c | p | Holm p |")
print("|---|---|---|---|---|---|---|")
res, pv = {}, {}
for name, d in others.items():
    ks = [k for k in H100 if k in d]
    if not ks:
        continue
    rows = [(A3[k]["image_id"], sc(A3[k]), sc(d[k])) for k in ks]
    bs = E.paired_bootstrap(rows)
    b = sum(1 for _, x, y in rows if x >= 0.5 > y)
    c = sum(1 for _, x, y in rows if y >= 0.5 > x)
    pv[name] = E.mcnemar_exact(b, c)
    res[name] = (len(ks), bs, b, c)
holm = E.holm(pv)
for name, (n, bs, b, c) in res.items():
    print(f"| {name} | {n} | {bs['diff']:+.3f} | {bs['ci_low']:+.3f} to {bs['ci_high']:+.3f} | {b}/{c} | "
          f"{pv[name]:.3g} | {holm[name]:.3g} |")

print("\nBy question type (accuracy):")
types = sorted({A3[k]["question_type"] for k in H100})
print("| Method | " + " | ".join(types) + " |")
print("|---|" + "---|" * len(types))
for name, d in [("A3v13", A3)] + list(others.items()):
    cells = []
    for t in types:
        ks = [k for k in H100 if k in d and A3[k]["question_type"] == t]
        cells.append(f"{sum(sc(d[k]) for k in ks) / len(ks):.3f}" if ks else "-")
    print(f"| {name} | " + " | ".join(cells) + " |")
