"""E8b: agreement between the two blinded evidence judges (Grok-4.3 = judge 1, GPT-4.1 = judge 2), and whether the
E8 conclusions hold under judge 2. Added post hoc (PROTOCOL.md, 2026-11-18): no coral expert was available.
"""
import gzip
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


def num(r, f):
    try:
        return float((r.get("rating") or {}).get(f))
    except (TypeError, ValueError, AttributeError):
        return None


def flag(r):
    v = (r.get("rating") or {}).get("contradicts_answer") if isinstance(r.get("rating"), dict) else None
    if isinstance(v, str):
        v = v.strip().lower() in {"true", "yes", "1"}
    return None if v is None else bool(v)


def kappa(pairs, cats, weights=None):
    """Cohen's kappa; weights='quadratic' for ordinal categories."""
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(pairs)
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1 / n
    pa = [sum(obs[i][j] for j in range(k)) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2 / (k - 1) ** 2 if weights == "quadratic" else float(i != j)) for j in range(k)] for i in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - do / de if de else float("nan")


def sc(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"]) >= 0.5


j1 = {"TRACE": load("E8_judge_A3v13"), "Agent": load("E8_judge_E7a")}
j2 = {"TRACE": load("E8b_judge2_A3v13"), "Agent": load("E8b_judge2_E7a")}
sysrec = {"TRACE": load("A3v13_trace"), "Agent": load("E7a_generic_gpt5_agent")}
keys = sorted(set(j1["TRACE"]) & set(j1["Agent"]) & set(j2["TRACE"]) & set(j2["Agent"]))
bad2 = sum(1 for s in j2 for r in j2[s].values() if not isinstance(r.get("rating"), dict))
print(f"Paired questions rated by both judges for both systems: {len(keys)}; judge-2 invalid ratings: {bad2}\n")

print("## Inter-judge agreement (1,200 traces)\n")
print("| Rating | Weighted κ (quadratic) | Exact agreement |")
print("|---|---|---|")
for f, cats in [("grounded", [0, 1, 2]), ("relevant", [0, 1, 2]), ("sufficient", [0, 1, 2]), ("overall", [1, 2, 3, 4, 5])]:
    pairs = []
    for s in ("TRACE", "Agent"):
        for k in keys:
            a, b = num(j1[s][k], f), num(j2[s][k], f)
            if a is not None and b is not None and int(a) in cats and int(b) in cats:
                pairs.append((int(a), int(b)))
    print(f"| {f} | {kappa(pairs, cats, 'quadratic'):.2f} | {sum(a == b for a, b in pairs) / len(pairs):.2f} |")
pairs = [(flag(j1[s][k]), flag(j2[s][k])) for s in ("TRACE", "Agent") for k in keys
         if flag(j1[s][k]) is not None and flag(j2[s][k]) is not None]
print(f"| contradicts_answer | {kappa(pairs, [False, True]):.2f} (unweighted) | "
      f"{sum(a == b for a, b in pairs) / len(pairs):.2f} |\n")


def err_stats(j, ks):
    out = {}
    for s in ("TRACE", "Agent"):
        ok = {k: sc(sysrec[s][k]) for k in ks}
        wrong = [k for k in ks if not ok[k]]
        ov = {k: num(j[s][k], "overall") for k in ks}
        pos = [ov[k] for k in ks if ok[k] and ov[k] is not None]
        neg = [ov[k] for k in ks if not ok[k] and ov[k] is not None]
        au = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
        vals = [ov[k] for k in ks if ov[k] is not None]   # list, not the dict: bootstrap samples repeat keys
        out[s] = {"mean overall": sum(vals) / len(vals),
                  "contradiction rate": sum(bool(flag(j[s][k])) for k in ks) / len(ks),
                  "convincing wrong": sum((ov[k] or 0) >= 4 for k in wrong) / len(wrong),
                  "wrong flagged": sum(bool(flag(j[s][k])) for k in wrong) / len(wrong),
                  "auroc": au}
    return out


img = {k: sysrec["TRACE"][k]["image_id"] for k in keys}
by = {}
for k in keys:
    by.setdefault(img[k], []).append(k)
ims = sorted(by)
print("## E8 conclusions under each judge (TRACE vs generic agent; 95% image-cluster CI of the difference, 2,000 resamples)\n")
print("| Measure | Judge | TRACE | Generic agent | Difference | 95% CI |")
print("|---|---|---|---|---|---|")
for label, j in (("Grok-4.3", j1), ("GPT-4.1", j2)):
    base = err_stats(j, keys)
    rng = random.Random(2026)
    boots = {m: [] for m in base["TRACE"]}
    for _ in range(2000):
        ks = [k for i in (rng.choice(ims) for _ in ims) for k in by[i]]
        s = err_stats(j, ks)
        for m in boots:
            boots[m].append(s["TRACE"][m] - s["Agent"][m])
    for m in base["TRACE"]:
        v = sorted(boots[m])
        print(f"| {m} | {label} | {base['TRACE'][m]:.3f} | {base['Agent'][m]:.3f} | "
              f"{base['TRACE'][m] - base['Agent'][m]:+.3f} | {v[50]:+.3f} to {v[1949]:+.3f} |")
