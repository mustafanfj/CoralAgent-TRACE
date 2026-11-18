"""E4: does giving the closed genus list help? (Reviewer comment C4.)

E4a: each baseline with and without the genus list on the H genus_taxonomy questions.
E4b: CoralAgent-TRACE (A3v13, which uses the list) vs A3v13 without the list on H100 genus questions.
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


def compare(name, with_list, without, keys):
    keys = [k for k in keys if k in with_list and k in without]
    rows = [(with_list[k]["image_id"], sc(with_list[k]), sc(without[k])) for k in keys]
    a = sum(x for _, x, _ in rows) / len(rows)
    b = sum(y for _, _, y in rows) / len(rows)
    bs = E.paired_bootstrap(rows)
    nb = sum(1 for _, x, y in rows if x >= 0.5 > y)
    nc = sum(1 for _, x, y in rows if y >= 0.5 > x)
    print(f"| {name} | {len(rows)} | {b:.3f} | {a:.3f} | {a - b:+.3f} | {bs['ci_low']:+.3f} to {bs['ci_high']:+.3f} | "
          f"{nb}/{nc} | {E.mcnemar_exact(nb, nc):.3g} |")


print("| Method | n | Without list | With list | Difference | 95% CI | McNemar b/c | p |")
print("|---|---|---|---|---|---|---|---|")
for label, base, gl in [("Strict GPT-5", "B0b_strict_v13", "B0b_strict_genuslist_v13"),
                        ("GPT-4.1", "B1_gpt41_v13", "B1_gpt41_genuslist_v13"),
                        ("Grok-4.3", "B2_grok_v13", "B2_grok_genuslist_v13")]:
    w, wo = load(gl), load(base)
    keys = [k for k, r in wo.items() if r["question_type"] == "genus_taxonomy"]
    compare(label, w, wo, keys)
a3, a3n = load("A3v13_trace"), load("A3v13_no_genus_list")
compare("CoralAgent-TRACE (H100 genus)", a3, a3n, list(a3n))
