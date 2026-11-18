"""Analyses that need no API calls (reviewer comments C2, C4, C5, C7, C8, C10, C13, plus cost per arm).

Run: python3 analysis/remaining_analyses.py > analysis/Remaining_analyses_output.md
Also writes analysis/evaluation_ids.csv (C4) and analysis/failure_cases.json (C13).
"""
import csv
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(ROOT, "results", "v13")


def load(m):
    p = os.path.join(R, f"{m}.jsonl.gz")
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(p))} if os.path.exists(p) else {}


def ok(r):
    return V.quick_semantic_score(r["question"], r["answer_gt"], r.get("answer_pred") or "", r["question_type"]) >= 0.5


A3, A5 = load("A3v13_trace"), load("A5v13_no_guard")
B0b, B1, B2 = load("B0b_strict_v13"), load("B1_gpt41_v13"), load("B2_grok_v13")
E5, E6, E7 = load("E5_single_call_same_info"), load("E6_self_consistency"), load("E7a_generic_gpt5_agent")
D = sorted(k for k, r in A3.items() if r["split"] == "D")
H = sorted(k for k, r in A3.items() if r["split"] == "H")
H100 = sorted(k for k in E6 if k in A3 and A3[k]["split"] == "H")


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


# ------------------------------------------------------------------ C4: data provenance and IDs
print("# Remaining analyses (no API calls)\n")
print("## C4. Evaluation data and identifiers\n")
print(f"- Questions evaluated: {len(A3)} on {len({r['image_id'] for r in A3.values()})} images. "
      f"D has {len(D)} questions and H has {len(H)}; H100 has {len(H100)}.")
print("- Source: `CoralVQA_train.jsonl`, the LLaVA-format **training** split of CoralVQA (Hugging Face "
      "`CoralReefData/CoralVQA`). No test or cross-region data were used.")
qt = Counter(r["question_type"] for r in A3.values())
print("- Question types (v12 classifier): " + ", ".join(f"{k} {v}" for k, v in sorted(qt.items())))
with open(os.path.join(HERE, "evaluation_ids.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "image_id", "split", "in_H100", "question_type", "visible_bleaching_task"])
    h100 = set(H100)
    for k in sorted(A3):
        r = A3[k]
        w.writerow([k, r["image_id"], r["split"], int(k in h100), r["question_type"],
                    int(E.in_visible_bleaching_task(r["question"], r["question_type"]))])
print("- All identifiers are in `analysis/evaluation_ids.csv`.\n")

# ------------------------------------------------------------------ C7: label conversion
print("## C7. Visible-bleaching labels\n")
groups = defaultdict(list)
for k, r in A3.items():
    q, t = r["question"], r["question_type"]
    if t != "bleaching_health":
        continue
    ql = q.lower()
    if E.in_visible_bleaching_task(q, t):
        g = "included: asks about visible bleaching"
    elif E.BLEACH.search(ql):
        g = "excluded: bleaching susceptibility or tendency"
    else:
        g = "excluded: other health (disease, mortality, condition, colour)"
    groups[g].append(k)
for g, ks in sorted(groups.items()):
    print(f"- {g}: {len(ks)} questions")
inc = groups["included: asks about visible bleaching"]
lab = Counter(E.label(A3[k]["question"], A3[k]["question_type"], A3[k]["answer_gt"]) for k in inc)
print(f"- Reference labels in the task: visible bleaching {lab['visible_bleaching']}, "
      f"no visible bleaching {lab['no_visible_bleaching']}, unmappable {lab[None]}.")
print("- v12 used one broad 'healthy/bleached' rule over every question type. Any answer containing white, pale or "
      "stressed counted as bleached; dead counted as bleached; susceptibility questions were included. That rule "
      "produced 776 binary questions, against 689 bleaching_health questions. The v13 task keeps only questions "
      "that ask whether bleaching is visible.")
rng = random.Random(2026)
print("\nExamples (seed 2026):\n")
for g, ks in sorted(groups.items()):
    print(f"*{g}*")
    for k in rng.sample(ks, min(3, len(ks))):
        q = re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", A3[k]["question"])
        print(f"  - \"{q.strip()}\" → reference \"{A3[k]['answer_gt']}\"")
print()

# ------------------------------------------------------------------ C8: per-class tables
print("## C8. Visible bleaching: per-class outcomes (correct / wrong / no verdict)\n")


def cls_table(name, d, keys):
    keys = [k for k in keys if k in d and E.in_visible_bleaching_task(d[k]["question"], d[k]["question_type"])]
    cnt = {c: Counter() for c in ("visible_bleaching", "no_visible_bleaching")}
    for k in keys:
        r = d[k]
        g = E.label(r["question"], r["question_type"], r["answer_gt"])
        if g is None:
            continue
        p = E.prediction(r["question"], r["question_type"], r.get("answer_pred"))
        cnt[g]["no verdict" if p is None else ("correct" if p == g else "wrong")] += 1
    v, n = cnt["visible_bleaching"], cnt["no_visible_bleaching"]
    nv, nn = sum(v.values()), sum(n.values())
    tp, fn_ = v["correct"], v["wrong"]
    tn, fp = n["correct"], n["wrong"]
    sens = tp / nv if nv else 0
    spec = tn / nn if nn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0
    cov = (nv + nn - v["no verdict"] - n["no verdict"]) / (nv + nn)
    ans_acc = (tp + tn) / max(1, tp + tn + fn_ + fp)
    print(f"| {name} | {v['correct']} / {v['wrong']} / {v['no verdict']} | {n['correct']} / {n['wrong']} / "
          f"{n['no verdict']} | {sens:.3f} | {spec:.3f} | {f1:.3f} | {cov:.3f} | {ans_acc:.3f} | "
          f"{v['no verdict'] + n['no verdict']} |")


hdr = ("| Method | Visible bleaching: correct / wrong / no verdict | No visible bleaching: correct / wrong / no verdict "
       "| Sensitivity | Specificity | F1 (visible) | Coverage | Answered-only accuracy | Referred (no verdict) |")
sep = "|---|---|---|---|---|---|---|---|---|"
for title, keys, arms in [("H (held-out, 400 images)", H, [("CoralAgent-TRACE", A3), ("TRACE before GUARD", A5),
                                                             ("Strict GPT-5", B0b), ("GPT-4.1", B1), ("Grok-4.3", B2)]),
                          ("D (development, 100 images)", D, [("CoralAgent-TRACE", A3), ("TRACE before GUARD", A5)]),
                          ("H100 (matched controls)", H100, [("CoralAgent-TRACE", A3), ("Generic GPT-5 agent", E7),
                                                             ("Single call, same info", E5),
                                                             ("Self-consistency k=3", E6), ("Strict GPT-5", B0b)])]:
    print(f"**{title}**\n")
    print(hdr)
    print(sep)
    for name, d in arms:
        cls_table(name, d, keys)
    print()
tech = sum(1 for r in A3.values() if r.get("error") not in (None, "", "None"))
empty = sum(1 for r in A3.values() if not str(r.get("answer_pred") or "").strip())
print(f"No-verdict cases are answers the mapping cannot read as yes/no or a count; they are not technical failures. "
      f"In v13, TRACE had {tech} technical failures (API or parse errors) and {empty} empty answers across "
      f"{len(A3)} questions. In v12, 444 of 3,000 answers were empty.\n")

# ------------------------------------------------------------------ C5: GUARD behaviour
print("## C5. What GUARD does (A3v13, all 3,000 questions)\n")
print("| Split | Question type | n | Verifier flags a problem | Repair attempted | Repair blocked by protection | "
      "Answer changed | Changed: wrong→right | Changed: right→wrong |")
print("|---|---|---|---|---|---|---|---|---|")
for split, keys in (("D", D), ("H", H)):
    by = defaultdict(list)
    for k in keys:
        by[A3[k]["question_type"]].append(k)
    for t in sorted(by) + ["all"]:
        ks = keys if t == "all" else by[t]
        uns = [k for k in ks if A3[k].get("verifier_supported") is False]
        rep = [k for k in ks if A3[k].get("verifier_repair_used")]
        ch = [k for k in ks if k in A5 and str(A3[k].get("answer_pred")).strip().lower()
              != str(A5[k].get("answer_pred")).strip().lower()]
        up = sum(1 for k in ch if ok(A3[k]) and not ok(A5[k]))
        down = sum(1 for k in ch if ok(A5[k]) and not ok(A3[k]))
        blocked = sum(1 for k in rep if "repair_blocked" in str(A3[k].get("finalizer_source")))
        flagged = sum(1 for k in ks if A3[k].get("verifier_repair_used") or A3[k].get("verifier_supported") is False)
        print(f"| {split} | {t} | {len(ks)} | {flagged} | {len(rep)} | {blocked} | {len(ch)} | {up} | {down} |")
print("\nAll other settings are identical: A5 is the same run read before verification, so its tools, prior and "
      "pre-verification answer match A3 exactly.\n")
lv = Counter(A3[k].get("verifier_support_level") for k in A3)
print("Final verifier verdict (after any repair and re-verification): "
      + ", ".join(f"{k}: {v}" for k, v in lv.most_common()) + ". 'Repair blocked by protection' means the verifier "
      "asked for a change but the existing answer was a valid answer of the requested type (genus name, colour, count, "
      "yes/no), so v12 protection kept it. Protected answers therefore change only when the pre-verification answer "
      "was empty or invalid for the question type.\n")

# GUARD examples
print("Examples of GUARD changes on H (seed 2026):\n")
ch_h = [k for k in H if k in A5 and str(A3[k].get("answer_pred")).strip().lower()
        != str(A5[k].get("answer_pred")).strip().lower()]
rng = random.Random(2026)
for k in rng.sample(ch_h, min(6, len(ch_h))):
    r = A3[k]
    v = r.get("verifier") or {}
    q = re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", r["question"]).strip()
    print(f"- [{r['question_type']}] \"{q}\" | reference \"{r['answer_gt']}\" | before GUARD \"{A5[k]['answer_pred']}\" "
          f"→ after \"{r['answer_pred']}\" ({'right' if ok(r) else 'wrong'}). Verifier: "
          f"{'; '.join(map(str, v.get('contradictions') or []))[:200] or v.get('support_level')}")
print()

# tool-call counting
print("Tool calls per question (mean; spatial_focus, baseline_prior, retrieval and every model-based tool each count "
      "as one call; GUARD and repair are not tool calls):\n")
print("| Configuration (D) | Mean tool calls |")
print("|---|---|")
for name, m in [("A3v13 full system", "A3v13_trace"), ("A0 SCOPE only", "A0v13_scope_only"),
                ("A1 prior + SCOPE", "A1v13_prior_scope"), ("A2 fixed all tools", "A2v13_fixed_all_tools"),
                ("A4 no retrieval", "A4v13_no_retrieval"), ("A6 no prior", "A6v13_no_prior"),
                ("A3 without SCOPE", "A3v13_no_scope")]:
    d = load(m)
    ks = [k for k in D if k in d]
    if ks:
        print(f"| {name} | {sum(int(d[k].get('n_tool_calls') or 0) for k in ks) / len(ks):.2f} |")
print()

# ------------------------------------------------------------------ C10: routing
print("## C10. Routing: question type versus keywords\n")
kw = [k for k, r in A3.items() if r["question_type"] in ("color_visual", "morphology_visual")
      and "genus" in r["question"].lower()]
gen_v13 = sum(1 for k in kw if "genus_identification" in (A3[k].get("tool_sequence") or []))
print(f"- Colour or morphology questions that mention 'genus': {len(kw)}.")
print(f"- v13 (answer-type priority, Eq. 5) sends {gen_v13} of them to genus_identification.")
# v12 routing, recomputed with the unchanged v12 functions on the same questions (no API calls)
import pandas as pd  # noqa: E402
v12_intent = V.question_intent
v12_plan = V.planned_tools_for_question
n12 = 0
for k in kw:
    row = pd.Series({"question": A3[k]["question"], "question_type": A3[k]["question_type"], "answer_gt": "",
                     "sample_id": k, "image_id": A3[k]["image_id"]})
    try:
        if "genus_identification" in v12_plan(row):
            n12 += 1
    except Exception as ex:  # pragma: no cover
        print(f"- v12 planner failed on {k}: {ex}")
        break
print(f"- The original v12 keyword routing sends {n12} of them to genus_identification "
      f"(the cause of Figure 9c/9d).")
acc_kw = sum(ok(A3[k]) for k in kw) / max(1, len(kw))
print(f"- v13 accuracy on these questions: {acc_kw:.3f}.\n")

# ------------------------------------------------------------------ C13: failure cases
print("## C13. Failure cases (selected at random with seed 2026 from each category, H split)\n")
cats = {
    "missed visible bleaching": [k for k in H if E.label(A3[k]["question"], A3[k]["question_type"], A3[k]["answer_gt"])
                                 == "visible_bleaching" and E.prediction(A3[k]["question"], A3[k]["question_type"],
                                                                         A3[k]["answer_pred"]) != "visible_bleaching"],
    "wrong genus": [k for k in H if A3[k]["question_type"] == "genus_taxonomy" and not ok(A3[k])],
    "wrong answer accepted by GUARD": [k for k in H if not ok(A3[k]) and A3[k].get("verifier_supported") is True],
    "GUARD changed a correct answer to a wrong one": [k for k in H if k in A5 and ok(A5[k]) and not ok(A3[k])],
    "verifier error contained by protection": [k for k in H if ok(A3[k]) and A3[k].get("verifier_repair_used")
                                               and "repair_blocked" in str(A3[k].get("finalizer_source"))],
}
rng = random.Random(2026)
cases = []
for c, ks in cats.items():
    print(f"- {c}: {len(ks)} cases on H")
    for k in rng.sample(ks, min(2, len(ks))):
        r = A3[k]
        cases.append({"category": c, "sample_id": k, "image_id": r["image_id"], "question": r["question"],
                      "reference": r["answer_gt"], "prior": r.get("baseline_prior_answer"),
                      "focused_vqa": r.get("focused_vqa_answer"), "genus_tool": r.get("genus_tool_answer"),
                      "before_guard": A5.get(k, {}).get("answer_pred"), "final": r["answer_pred"],
                      "finalizer_source": r.get("finalizer_source"), "tools": r.get("tool_sequence"),
                      "verifier": r.get("verifier"), "repair_used": r.get("verifier_repair_used"),
                      "region": (r.get("observations") or [{}])[0].get("result", {}).get("region_label")})
json.dump(cases, open(os.path.join(HERE, "failure_cases.json"), "w"), indent=1, default=str)
print("\n| Category | Question | Reference | Prior / focused VQA | Before GUARD → final | Verifier | Likely stage |")
print("|---|---|---|---|---|---|---|")
for c in cases:
    q = re.sub(r"^\[vqa\]\s*|\s*Provide your answer as short as possible\.?", "", c["question"]).strip()
    v = c["verifier"] or {}
    if c["category"].startswith("GUARD changed"):
        stage = "GUARD repair"
    elif str(c["prior"]).lower() == str(c["final"]).lower() and str(c["focused_vqa"]).lower() != str(c["final"]).lower():
        stage = "arbitration followed the prior"
    elif str(c["focused_vqa"]).lower() == str(c["final"]).lower():
        stage = "focused VQA perception"
    else:
        stage = "synthesis / tools disagree"
    if c["category"].startswith("verifier error"):
        stage = "verifier false alarm (no effect)"
    print(f"| {c['category']} | {q[:110]} | {c['reference']} | {c['prior']} / {c['focused_vqa']} | "
          f"{c['before_guard']} → {c['final']} | {v.get('support_level')} | {stage} |")
print("\nFull records are in `analysis/failure_cases.json`.\n")

# ------------------------------------------------------------------ cost per arm
print("## Cost and model calls per question (batch prices, from logged usage)\n")
use = json.load(open(os.path.join(HERE, "usage_by_stage.json")))["usage_by_stage"]


def u(stage, prov, model):
    v = use.get(f"{stage}|{prov}|{model}") or {}
    return v.get("responses", 0), v.get("cost_usd", 0.0)


rows = [
    ("Strict GPT-5", [("E1", "openai", "gpt-5")], 2400),
    ("GPT-4.1", [("E1", "openai", "gpt-4.1")], 2400),
    ("Grok-4.3", [("E1", "xai", "grok-4.3")], 2400),
    ("CoralAgent-TRACE (E2, plus the strict GPT-5 prior it reuses)", [("E2", "openai", "gpt-5"), ("E2", "xai", "grok-4.3")], 2950),
    ("Single call, same information (E5)", [("E5", "openai", "gpt-5")], 600),
    ("Self-consistency k=3 (E6; 2 new samples + the strict answer)", [("E6", "openai", "gpt-5")], 600),
    ("Generic GPT-5 agent with GUARD (E7a)", [("E7a", "openai", "gpt-5"), ("E7a", "xai", "grok-4.3")], 600),
]
print("| Method | Model calls per question | Cost per question (USD) |")
print("|---|---|---|")
strict_c = u("E1", "openai", "gpt-5")[1] / 2400
for name, parts, n in rows:
    calls = sum(u(*p)[0] for p in parts) / n
    cost = sum(u(*p)[1] for p in parts) / n
    if name.startswith("CoralAgent") or name.startswith("Self-consistency"):
        calls += 1
        cost += strict_c
    print(f"| {name} | {calls:.2f} | {cost:.4f} |")
e7_tools = sum(len(E7[k].get("tool_sequence") or []) for k in H100) / len(H100)
e7_turns = sum(len(E7[k].get("agent_turn_keys") or []) for k in H100) / len(H100)
a3_tools = sum(int(A3[k].get("n_tool_calls") or 0) for k in H100) / len(H100)
print(f"\nOn H100: TRACE makes {a3_tools:.2f} tool calls per question in one planned pass. The generic agent makes "
      f"{e7_tools:.2f} tool calls over {e7_turns:.2f} sequential planning turns.")
print("Latency: all runs used provider batch APIs, so wall-clock times reflect batch queues, not inference. Sequential "
      "model rounds per question indicate the latency in interactive use. TRACE needs one parallel tool round, then GUARD and "
      "at most one repair and re-verification. The generic agent needs one round per planning turn, then GUARD.")
