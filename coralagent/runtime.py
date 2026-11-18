"""Load the frozen v12 core, wire it to the batch gateway, and apply the v13 fixes.

v13 fixes (pre-registered, PROTOCOL.md section 1):
  1. Output budget: GPT-5 tool/finalizer/repair calls use reasoning effort 'low' and >= 2000 output tokens.
  2. Baseline prior: strict GPT-5 zero-shot answers (B0b), not the deprecated B0 run.
  3. Answer-type priority: a question's CoralVQA category (color / morphology) decides the route before a
     'genus' keyword can (matches Eq. 5); GUARD receives the matching expected answer style.
  4. Ablations are built on A3 (see tasks.py).
Nothing else in the v12 logic is changed.
"""
import contextlib
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config, gateway

config.ensure_dirs()
os.environ.setdefault("CORALAGENT_WORK_DIR", str(config.WORK))
os.environ.setdefault("CORALAGENT_SEND_TEMPERATURE", "0")
from . import v12_core as core  # noqa: E402

core.FOCUS_CROP_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------
# Context for the variant being run (cache namespace, region override, genus list)
# ----------------------------------------------------------------------------------
CTX: Dict[str, Any] = {"ns": "base", "region_override": None, "genus_list": True}
_FULL_GENUS_LIST: List[str] = []
PRIOR: Dict[str, str] = {}
PRIOR_SOURCE = "B0b_gpt5_zero_shot_strict"


@contextlib.contextmanager
def variant(ns: str = "base", region_override: Optional[str] = None, genus_list: bool = True):
    old = dict(CTX)
    CTX.update({"ns": ns, "region_override": region_override, "genus_list": genus_list})
    core.KNOWN_GENUS_LIST = list(_FULL_GENUS_LIST) if genus_list else []
    core.GENUS_CONSTRAIN_TO_KNOWN = bool(genus_list)
    try:
        yield
    finally:
        CTX.clear(); CTX.update(old)
        core.KNOWN_GENUS_LIST = list(_FULL_GENUS_LIST) if old["genus_list"] else []
        core.GENUS_CONSTRAIN_TO_KNOWN = bool(old["genus_list"])


# ----------------------------------------------------------------------------------
# Gateway wiring: every v12 model call becomes a batch request
# ----------------------------------------------------------------------------------
ROLE_PROVIDER = {"tool": "openai", "agent": "openai", "baseline": "openai", "verifier": "xai", "judge": "xai"}
ROLE_MODEL = {"openai": config.GPT5, "xai": config.GROK, "together": config.GEMMA}
_ROLE = {"name": "direct"}


def _v13_call_llm(provider, model, prompt, instructions="", image_path=None, expect_json=False,
                  max_tokens=900, temperature=None):
    purpose = _ROLE["name"]
    effort = None
    if provider == "openai" and str(model).startswith("gpt-5") and purpose in {"tool", "agent"}:
        effort = config.GPT5_V13_EFFORT                                     # v13 fix 1
        max_tokens = max(int(max_tokens), config.GPT5_V13_MIN_OUTPUT)
    spec = gateway.make_spec(provider, model, prompt, instructions, [image_path] if image_path else [],
                             expect_json, max_tokens, purpose=purpose, effort=effort)
    resp = gateway.lookup(spec)
    return gateway.as_v12_result(spec, resp)


def _v13_call_vlm(role, prompt, instructions="", image_path=None, expect_json=False, max_tokens=900,
                  temperature=None):
    provider = ROLE_PROVIDER.get(role, "openai")     # no cross-provider fallback in v13
    old = _ROLE["name"]
    _ROLE["name"] = role
    try:
        out = _v13_call_llm(provider, ROLE_MODEL[provider], prompt, instructions, image_path, expect_json,
                            max_tokens, temperature)
    finally:
        _ROLE["name"] = old
    out["role"] = role
    return out


def _direct_call_llm(provider, model, prompt, instructions="", image_path=None, expect_json=False,
                     max_tokens=900, temperature=None):
    # Direct v12 call_llm calls in the TRACE path are the GUARD verifier calls (xAI).
    old = _ROLE["name"]
    if _ROLE["name"] == "direct":
        _ROLE["name"] = "verifier" if provider == "xai" else "direct"
    try:
        return _v13_call_llm(provider, model, prompt, instructions, image_path, expect_json, max_tokens, temperature)
    finally:
        _ROLE["name"] = old


_v12_parse_json_loose = core.parse_json_loose
_BARE_VALUE = re.compile(r'("[^"\n]{1,60}"\s*:\s*)(?!\s|["{\[\-\d]|true\b|false\b|null\b)([^,\n{}\[\]"]+?)(\s*(?:,|\n|\}))')


def _v13_parse_json_loose(text):
    """v12 parser plus one repair: bare (unquoted) string values such as {"answer": Acropora}. GPT-4.1 wrote
    these in 207/349 E4 genus-list answers. Only output that v12 could not parse is affected."""
    out = _v12_parse_json_loose(text)
    if not (isinstance(out, dict) and out.get("parse_error")) or not isinstance(text, str):
        return out
    fixed = _BARE_VALUE.sub(lambda m: m.group(1) + json.dumps(m.group(2).strip()) + m.group(3),
                            core.clean_json_text(text))
    repaired = _v12_parse_json_loose(fixed)
    return repaired if isinstance(repaired, dict) and not repaired.get("parse_error") else out


core.parse_json_loose = _v13_parse_json_loose
core.call_llm = _direct_call_llm
core.call_vlm = _v13_call_vlm


# ----------------------------------------------------------------------------------
# Tool cache: namespaced per variant; persisted by the runner, not on every write
# ----------------------------------------------------------------------------------
core.TOOL_CACHE = {}
core.save_tool_cache = lambda cache: None
core.cache_key = lambda row, tool_name: f"{CTX['ns']}::{row['sample_id']}::{tool_name}"


# ----------------------------------------------------------------------------------
# Region override (no-SCOPE ablation and the generic agent's region argument)
# ----------------------------------------------------------------------------------
REGIONS = ["whole_image", "top_left", "top_center", "top_right", "center_left", "center_center",
           "center_right", "bottom_left", "bottom_center", "bottom_right"]


def region_metadata(row, region_label: str) -> Dict[str, Any]:
    if region_label == "whole_image":
        return {"region_label": "whole_image", "bbox_norm": [0.0, 0.0, 1.0, 1.0], "uses_crop": False,
                "focus_image_path": str(row["local_image_path"]), "original_image_path": str(row["local_image_path"])}
    v, h = region_label.split("_")
    y_map = {"top": (0.0, 0.48), "center": (0.22, 0.78), "bottom": (0.52, 1.0)}
    x_map = {"left": (0.0, 0.48), "center": (0.22, 0.78), "right": (0.52, 1.0)}
    y0, y1 = y_map[v]; x0, x1 = x_map[h]
    safe_id = re.sub("[^A-Za-z0-9_.-]+", "_", str(row["sample_id"]))[:180]
    out_path = core.FOCUS_CROP_DIR / f"focus_{safe_id}_{region_label}.jpg"   # same naming as v12
    if not out_path.exists():
        img = core.Image.open(row["local_image_path"]).convert("RGB")
        img = core.ImageOps.exif_transpose(img)
        w, hh = img.size
        pad = 0.04
        box = (int(max(0.0, x0 - pad) * w), int(max(0.0, y0 - pad) * hh), int(min(1.0, x1 + pad) * w),
               int(min(1.0, y1 + pad) * hh))
        img.crop(box).save(out_path, quality=92)
    return {"region_label": region_label, "bbox_norm": [x0, y0, x1, y1], "uses_crop": True,
            "focus_image_path": str(out_path), "original_image_path": str(row["local_image_path"])}


def _v13_tool_spatial_focus(row):
    cached = core.get_cached(row, "spatial_focus")
    if cached:
        return cached
    if CTX["region_override"]:
        meta = region_metadata(row, CTX["region_override"])
    else:
        meta = core.focus_metadata(row)
    return core.set_cached(row, "spatial_focus", meta)


core.tool_spatial_focus = _v13_tool_spatial_focus


# ----------------------------------------------------------------------------------
# v13 fix 2: baseline prior = strict GPT-5 zero-shot answer
# ----------------------------------------------------------------------------------
def _v13_tool_baseline_prior(row):
    cached = core.get_cached(row, "baseline_prior")
    if cached:
        return cached
    sid = str(row["sample_id"])
    if sid not in PRIOR:
        raise gateway.WaitDependency(f"strict GPT-5 prior missing for {sid}")
    result = {"answer": core.extract_answer_field(PRIOR[sid]), "source": "baseline_checkpoint",
              "baseline_method": PRIOR_SOURCE, "provider": "openai", "model": config.GPT5,
              "confidence": None, "error": None}
    return core.set_cached(row, "baseline_prior", result)


core.tool_baseline_prior = _v13_tool_baseline_prior


# ----------------------------------------------------------------------------------
# v13 fix 3: the CoralVQA category decides the answer type before a 'genus' keyword
# ----------------------------------------------------------------------------------
_v12_question_intent = core.question_intent


def _v13_question_intent(row):
    it = dict(_v12_question_intent(row))
    qt = it["question_type"]
    if qt in {"color_visual", "morphology_visual"} and not it["asks_greatest_area"]:
        it["asks_genus"] = False
    style_by_type = {"color_visual": "color_word", "morphology_visual": "morphology_term",
                     "genus_taxonomy": "genus_name"}
    if qt in style_by_type:
        it["answer_style"] = style_by_type[qt]
    elif not it["asks_genus"] and it["answer_style"] == "genus_name":
        it["answer_style"] = "short_phrase"
    return it


core.question_intent = _v13_question_intent
v12_question_intent = _v12_question_intent


# ----------------------------------------------------------------------------------
# Tools of one plan are independent: discover all missing tool calls in one pass
# ----------------------------------------------------------------------------------
def _v13_run_trace_variant_once(row, cfg):
    observations, trajectory, missing = [], [], []
    intent = core.question_intent(row)
    plan = core.planned_tools_for_variant(row, cfg)
    for step, tool_name in enumerate(plan, start=1):
        try:
            result = core.execute_tool(tool_name, row, {"query": row.get("question", "")})
        except gateway.NeedCall as e:
            missing.append(str(e)); continue
        obs = {"step": step, "tool": tool_name, "result": result}
        observations.append(obs)
        trajectory.append({"step": step, "decision": {"action": "call_tool", "tool_name": tool_name,
                           "reason": f"{cfg['method']}_policy", "intent": intent, "plan": plan}, "observation": obs})
    if missing:
        raise gateway.NeedCall(",".join(missing))
    final = core.finalize_agent_answer(row, observations)
    if cfg.get("kind") == "scope_only":
        fv = core._obs_by_tool(observations).get("focused_vqa", {})
        final.update({"answer": core.canonicalize_for_intent(row, fv.get("answer", final.get("answer", "Unclear"))),
                      "finalizer_source": "scope_only_focused_vqa"})
    elif cfg.get("kind") == "prior_scope":
        final["finalizer_source"] = "prior_scope_consensus" if str(final.get("finalizer_source", "")).strip() else "prior_scope"
    trajectory.append({"step": len(trajectory) + 1, "decision": {"action": "answer", "intent": intent, "plan": plan,
                                                                  "ablation": cfg["method"]}, "final_response": final})
    return {"final": final, "observations": observations, "trajectory": trajectory, "intent": intent, "plan": plan}


core.run_trace_variant_once = _v13_run_trace_variant_once


# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
BANK: Optional[pd.DataFrame] = None


def load_bank(images_dir: Path = config.IMAGES) -> pd.DataFrame:
    global BANK
    rows = list(csv.DictReader(open(config.BANK_CSV, encoding="utf-8")))
    df = pd.DataFrame(rows)
    df["active_image_index"] = df["active_image_index"].astype(int)
    df["local_image_path"] = [str(images_dir / str(p).replace("\\", "/").split("/")[-1]) for p in df["local_image_path"]]
    fb = os.environ.get("CORAL_TEST_IMAGE_FALLBACK")          # offline tests only
    if fb:
        df["local_image_path"] = [p if Path(p).exists() else fb for p in df["local_image_path"]]
    df["gt_binary"] = df["gt_binary"].replace({"": None})
    df["split"] = ["D" if i < 100 else "H" for i in df["active_image_index"]]
    BANK = df
    return df


def load_genus_list() -> List[str]:
    global _FULL_GENUS_LIST
    raw = [json.loads(l) for l in open(config.CORALVQA_JSONL, encoding="utf-8") if l.strip()]
    vqa_df = core.normalize_rows(raw)
    labels = sorted({str(x).strip() for x in vqa_df.loc[vqa_df["question_type"].eq("genus_taxonomy"), "answer_gt"].dropna()
                     if core._looks_like_label_answer(x)})
    labels = [g for g in labels if len(g.split()) <= 3 and g.lower() not in {"hard", "soft", "coral", "fish"}]
    _FULL_GENUS_LIST = labels
    core.KNOWN_GENUS_LIST = list(labels)
    return labels


def load_prior(extra_rows: Optional[List[Dict[str, Any]]] = None) -> int:
    """Strict GPT-5 answers: v12 B0b rows for images 1-100 plus v13 E1 rows for images 101-500."""
    path = config.V12_BASELINES / "B0b_gpt5_zero_shot_strict_v12_TRACE_bank500_q6.jsonl"
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if not r.get("error") and str(r.get("answer_pred") or "").strip():
                PRIOR[str(r["sample_id"])] = r["answer_pred"]
    for r in extra_rows or []:
        if not r.get("error") and str(r.get("answer_pred") or "").strip():
            PRIOR[str(r["sample_id"])] = r["answer_pred"]
    return len(PRIOR)


def init(images_dir: Path = config.IMAGES) -> pd.DataFrame:
    config.ensure_dirs()
    core.FOCUS_CROP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_bank(images_dir)
    load_genus_list()
    load_prior()
    return df
