"""E8: blinded evidence-quality judge (Grok-4.3) on H100, CoralAgent-TRACE (A3v13) vs the generic GPT-5 agent (E7a).

Paired by question. Ordinal ratings: mean difference with a paired image-cluster bootstrap CI and a
sign test (Wilcoxon-style counts). contradicts_answer: exact McNemar.
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


ja, je = load("E8_judge_A3v13"), load("E8_judge_E7a")
a3, e7 = load("A3v13_trace"), load("E7a_generic_gpt5_agent")
keys = sorted(k for k in ja if k in je and isinstance(ja[k]["rating"], dict) and isinstance(je[k]["rating"], dict))
print(f"Paired questions with valid ratings from both: {len(keys)} "
      f"(invalid: A3 {sum(not isinstance(r['rating'], dict) for r in ja.values())}, "
      f"E7a {sum(not isinstance(r['rating'], dict) for r in je.values())})\n")


def num(r, f):
    v = r["rating"].get(f)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


print("| Rating | CoralAgent-TRACE | Generic GPT-5 agent | Difference | 95% CI | better/worse | sign-test p |")
print("|---|---|---|---|---|---|---|")
for f in ["grounded", "relevant", "sufficient", "overall"]:
    rows = [(a3[k]["image_id"], num(ja[k], f), num(je[k], f)) for k in keys]
    rows = [r for r in rows if r[1] is not None and r[2] is not None]
    bs = E.paired_bootstrap(rows)
    b = sum(1 for _, x, y in rows if x > y)
    c = sum(1 for _, x, y in rows if y > x)
    ma = sum(x for _, x, _ in rows) / len(rows)
    me = sum(y for _, _, y in rows) / len(rows)
    print(f"| {f} | {ma:.2f} | {me:.2f} | {bs['diff']:+.2f} | {bs['ci_low']:+.2f} to {bs['ci_high']:+.2f} | "
          f"{b}/{c} | {E.mcnemar_exact(b, c):.3g} |")

ca = {k: bool(ja[k]["rating"].get("contradicts_answer")) for k in keys}
ce = {k: bool(je[k]["rating"].get("contradicts_answer")) for k in keys}
b = sum(1 for k in keys if ca[k] and not ce[k])
c = sum(1 for k in keys if ce[k] and not ca[k])
print(f"| contradicts_answer (rate) | {sum(ca.values()) / len(keys):.3f} | {sum(ce.values()) / len(keys):.3f} | "
      f"{(sum(ca.values()) - sum(ce.values())) / len(keys):+.3f} | | {b}/{c} | {E.mcnemar_exact(b, c):.3g} |")


def sc(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"])


print("\nDoes the judge's overall score track correctness? (mean overall rating)")
print("| System | correct answers | wrong answers |")
print("|---|---|---|")
for name, j, s in [("CoralAgent-TRACE", ja, a3), ("Generic GPT-5 agent", je, e7)]:
    good = [num(j[k], "overall") for k in keys if sc(s[k]) >= 0.5 and num(j[k], "overall") is not None]
    bad = [num(j[k], "overall") for k in keys if sc(s[k]) < 0.5 and num(j[k], "overall") is not None]
    print(f"| {name} | {sum(good) / len(good):.2f} (n={len(good)}) | {sum(bad) / len(bad):.2f} (n={len(bad)}) |")


# ---------------------------------------------------------------------------
# Post-hoc (not pre-registered): is the evidence trace useful for catching errors?
# ---------------------------------------------------------------------------
import random  # noqa: E402

correct = {"TRACE": {k: sc(a3[k]) >= 0.5 for k in keys}, "Agent": {k: sc(e7[k]) >= 0.5 for k in keys}}
judged = {"TRACE": ja, "Agent": je}


def auroc(scores, labels):
    pos = [s for s, c in zip(scores, labels) if c]
    neg = [s for s, c in zip(scores, labels) if not c]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def err_stats(ks):
    out = {}
    for name in ("TRACE", "Agent"):
        j, c = judged[name], correct[name]
        ov = [num(j[k], "overall") for k in ks]
        lab = [c[k] for k in ks]
        flag = [bool(j[k]["rating"].get("contradicts_answer")) for k in ks]
        wrong = [i for i, x in enumerate(lab) if not x]
        out[name] = {
            "auroc": auroc(ov, lab),
            "convincing_wrong": sum(ov[i] >= 4 for i in wrong) / len(wrong),
            "wrong_flagged": sum(flag[i] for i in wrong) / len(wrong),
        }
    return out


base = err_stats(keys)
by_img = {}
for k in keys:
    by_img.setdefault(a3[k]["image_id"], []).append(k)
imgs = sorted(by_img)
rng = random.Random(2026)
boot = {m: [] for m in base["TRACE"]}
for _ in range(2000):
    ks = [k for i in (rng.choice(imgs) for _ in imgs) for k in by_img[i]]
    s = err_stats(ks)
    for m in boot:
        boot[m].append(s["TRACE"][m] - s["Agent"][m])
labels = {"auroc": "AUROC of the judge's overall rating for answer correctness",
          "convincing_wrong": "Wrong answers backed by convincing evidence (overall >= 4)",
          "wrong_flagged": "Wrong answers whose trace shows a contradiction (error flagged)"}
print("\nPost hoc: does the evidence trace expose errors? (paired image-cluster bootstrap, 2,000 resamples)")
print("| Measure | CoralAgent-TRACE | Generic GPT-5 agent | Difference | 95% CI |")
print("|---|---|---|---|---|")
for m, lab in labels.items():
    v = sorted(boot[m])
    print(f"| {lab} | {base['TRACE'][m]:.3f} | {base['Agent'][m]:.3f} | {base['TRACE'][m] - base['Agent'][m]:+.3f} | "
          f"{v[50]:+.3f} to {v[1949]:+.3f} |")

print("\nPost hoc: accuracy when only the answers with the most trusted evidence are kept (matched coverage)")
print("| Coverage | CoralAgent-TRACE | Generic GPT-5 agent |")
print("|---|---|---|")
ranked = {n: sorted(keys, key=lambda k: (bool(judged[n][k]["rating"].get("contradicts_answer")),
                                         -num(judged[n][k], "overall"), k)) for n in judged}
for cov in (0.5, 0.6, 0.7, 0.8, 1.0):
    cells = []
    for n in ("TRACE", "Agent"):
        ks = ranked[n][:int(len(keys) * cov)]
        cells.append(f"{sum(correct[n][k] for k in ks) / len(ks):.3f}")
    print(f"| {int(cov * 100)}% | " + " | ".join(cells) + " |")
