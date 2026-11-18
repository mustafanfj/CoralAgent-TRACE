import gzip, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from coralagent import v12_core as V
from coralagent import evaluation as E
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "v13")


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(f"{R}/{m}.jsonl.gz"))}


def sc(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"])


A3, A5 = load("A3v13_trace"), load("A5v13_no_guard")
B = {m: load(m) for m in ["B0b_strict_v13", "B1_gpt41_v13", "B2_grok_v13"]}
for split in "DH":
    for name, d in [("A3v13", A3), ("A5 noGUARD", A5)]:
        rows = [r for r in d.values() if r["split"] == split]
        print(f"{split} {name}: n={len(rows)} acc={sum(map(sc, rows)) / len(rows):.4f}")
H = sorted(k for k, r in A3.items() if r["split"] == "H")
a = {k: sc(A3[k]) for k in H}
bt = defaultdict(list)
for k in H:
    bt[A3[k]["question_type"]].append(a[k])
print("A3 H by type:", "  ".join(f"{t}={sum(v) / len(v):.3f}" for t, v in sorted(bt.items())))
print("errors A3:", sum(1 for r in A3.values() if r.get("error") not in (None, "None", "")))
print([n for n in dir(E) if not n.startswith("_")])

from collections import Counter
print("score values:", Counter(round(v, 2) for v in a.values()).most_common(6))
bin_ = lambda x: 1 if x >= 0.5 else 0
for ref, d in [("strict GPT-5", B["B0b_strict_v13"]), ("GPT-4.1", B["B1_gpt41_v13"]), ("Grok", B["B2_grok_v13"]), ("A5 noGUARD", A5)]:
    rows = [(A3[k]["image_id"], a[k], sc(d[k])) for k in H]
    bs = E.paired_bootstrap(rows)
    b = sum(1 for _, x, y in rows if bin_(x) and not bin_(y)); c = sum(1 for _, x, y in rows if not bin_(x) and bin_(y))
    print(f"A3v13 - {ref}: {bs}  McNemar b={b} c={c} p={E.mcnemar_exact(b, c):.4f}")
