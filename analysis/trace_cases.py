"""Select the held-out cases shown in the qualitative-trace figure (same four categories as the submitted Figure 9).

Selection is random with seed 2026 within each category, on the held-out split. Writes analysis/trace_cases.json,
which the lab-PC stage IMG reads to export the matching (downscaled) images to reports/trace_images/.
"""
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from coralagent import evaluation as E  # noqa: E402
from coralagent import v12_core as V  # noqa: E402

R = os.path.join(HERE, "..", "results", "v13")


def load(m):
    return {r["sample_id"]: r for r in map(json.loads, gzip.open(os.path.join(R, f"{m}.jsonl.gz")))}


def ok_ans(r, a):
    return V.quick_semantic_score(r["question"], r["answer_gt"], a or "", r["question_type"]) >= 0.5


A3, A5 = load("A3v13_trace"), load("A5v13_no_guard")
H = sorted(k for k, r in A3.items() if r["split"] == "H")


def bbox(r):
    for o in r.get("observations") or []:
        if o.get("tool") == "spatial_focus":
            res = o.get("result") or {}
            return res.get("region_label"), res.get("bbox_norm"), res.get("uses_crop")
    return None, None, None


def lab(r):
    return E.label(r["question"], r["question_type"], r["answer_gt"])


def pred(r):
    return E.prediction(r["question"], r["question_type"], r["answer_pred"])


cats = {
    "a": ("Focused evidence overturns an incorrect bleaching prior",
          [k for k in H if A3[k]["question_type"] == "bleaching_health" and ok_ans(A3[k], A3[k]["answer_pred"])
           and A3[k].get("baseline_prior_answer") and not ok_ans(A3[k], A3[k]["baseline_prior_answer"])
           and E.in_visible_bleaching_task(A3[k]["question"], A3[k]["question_type"])]),
    "b": ("Visible bleaching detected and supported by GUARD",
          [k for k in H if E.in_visible_bleaching_task(A3[k]["question"], A3[k]["question_type"])
           and lab(A3[k]) == "visible_bleaching" and pred(A3[k]) == "visible_bleaching"
           and A3[k].get("verifier_supported") is True]),
    "c": ("SCOPE crop gives the correct region-specific answer",
          [k for k in H if bbox(A3[k])[2] and A3[k]["question_type"] != "bleaching_health"
           and ok_ans(A3[k], A3[k]["answer_pred"]) and ok_ans(A3[k], A3[k].get("focused_vqa_answer"))
           and A3[k].get("baseline_prior_answer") and not ok_ans(A3[k], A3[k]["baseline_prior_answer"])]),
    "d": ("Correct answer protected from a verifier error",
          [k for k in H if ok_ans(A3[k], A3[k]["answer_pred"]) and A3[k].get("verifier_repair_used")
           and "repair_blocked" in str(A3[k].get("finalizer_source"))]),
}
rng = random.Random(2026)
out = []
for key, (title, ks) in cats.items():
    k = rng.choice(ks)
    r = A3[k]
    region, bb, crop = bbox(r)
    v = r.get("verifier") or {}
    out.append({"panel": key, "title": title, "n_in_category": len(ks), "sample_id": k, "image_id": r["image_id"],
                "question": r["question"], "reference": r["answer_gt"], "prior": r.get("baseline_prior_answer"),
                "focused_vqa": r.get("focused_vqa_answer"), "before_guard": A5.get(k, {}).get("answer_pred"),
                "final": r["answer_pred"], "tools": r.get("tool_sequence"), "region": region, "bbox_norm": bb,
                "uses_crop": crop, "verifier_level": v.get("support_level"),
                "verifier_suggested": v.get("suggested_answer"), "finalizer_source": r.get("finalizer_source")})
    print(key, len(ks), k, r["question"][:90], "| ref", r["answer_gt"], "| prior", r.get("baseline_prior_answer"),
          "| fvqa", r.get("focused_vqa_answer"), "| final", r["answer_pred"], "| region", region, "| ver",
          v.get("support_level"), v.get("suggested_answer"))
fails = json.load(open(os.path.join(HERE, "failure_cases.json")))
json.dump({"figure_cases": out, "extra_image_ids": sorted({c["image_id"] for c in fails})},
          open(os.path.join(HERE, "trace_cases.json"), "w"), indent=1, default=str)
