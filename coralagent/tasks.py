"""Per-row task functions for every arm (method) of the pre-registered experiments.

Each function takes a bank row and returns a result record, or raises gateway.NeedCall /
gateway.WaitDependency while responses or dependencies are still missing.
"""
import json
from typing import Any, Dict, List, Optional

from . import config, gateway, runtime
from .runtime import core

# =========================== zero-shot baselines ================================


def _genus_line() -> str:
    return "Possible genus labels (choose one exact label when the answer is a genus): " + \
        ", ".join(core.KNOWN_GENUS_LIST[:90])


def strict_gpt5(row, method: str, sample_idx: int = 0, genus_list: bool = False) -> Dict[str, Any]:
    """Strict GPT-5 baseline: v12 prompt unchanged. Output budget as for every v13 GPT-5 call (deviation 3,
    PROTOCOL.md): with v12's 120-token cap and default effort, 9/10 E1 test answers were empty."""
    prompt = f"Question: {row['question']}\nAnswer only."
    if genus_list:
        prompt += "\n" + _genus_line()
    spec = gateway.make_spec("openai", config.GPT5, prompt, core.STRICT_GPT5_SYSTEM, [row["local_image_path"]],
                             expect_json=False, max_tokens=config.GPT5_V13_MIN_OUTPUT, purpose="baseline_strict",
                             sample_idx=sample_idx, effort=config.GPT5_V13_EFFORT)
    resp = gateway.lookup(spec)
    r = gateway.as_v12_result(spec, resp)
    ans = core.extract_answer_field(r.get("text", ""))
    return _baseline_record(row, method, ans, {"answer": ans}, None, None, r)


def reefbank(row, method: str, provider: str, model: str, genus_list: bool = False) -> Dict[str, Any]:
    """v12 ReefBank zero-shot baseline prompt (JSON answer), used for Grok, Gemma and GPT-4.1."""
    prompt = f"Question: {row['question']}\nReturn a concise CoralVQA-style answer."
    if genus_list:
        prompt += "\n" + _genus_line()
    spec = gateway.make_spec(provider, model, prompt, core.REEFBANK_SYSTEM, [row["local_image_path"]],
                             expect_json=True, max_tokens=550, purpose="baseline_reefbank")
    resp = gateway.lookup(spec)
    r = gateway.as_v12_result(spec, resp)
    pred = r["json"] if isinstance(r.get("json"), dict) else {}
    ans = core.extract_answer_field(pred.get("answer", r.get("text", "")), fallback=r.get("text", ""))
    return _baseline_record(row, method, ans, pred, pred.get("health_status"), pred.get("confidence"), r)


def _baseline_record(row, method, ans, pred, health, conf, r) -> Dict[str, Any]:
    return {"sample_id": row["sample_id"], "image_id": row["image_id"], "split": row["split"],
            "question": row["question"], "answer_gt": row["answer_gt"], "question_type": row["question_type"],
            "gt_binary": row.get("gt_binary"), "method": method, "prediction": pred, "answer_pred": ans,
            "health_status_pred": health, "confidence": conf, "provider": r["provider"], "model": r["model"],
            "model_returned": r.get("model_returned"), "usage": r.get("usage"), "error": r.get("error"),
            "request_keys": [r.get("request_key")]}


# =========================== CoralAgent-TRACE variants =============================
V13_SPECS = {
    # proposed system
    "A3v13_trace": {"method": "A3v13_trace", "kind": "trace", "disable_bleaching_tools": True, "verifier_mode": "grok"},
    # ablations, all built on A3 (one component removed / added)
    "A0v13_scope_only": {"method": "A0v13_scope_only", "kind": "scope_only", "verifier_mode": "none"},
    "A1v13_prior_scope": {"method": "A1v13_prior_scope", "kind": "prior_scope", "verifier_mode": "none"},
    "A2v13_fixed_all_tools": {"method": "A2v13_fixed_all_tools", "kind": "trace", "fixed_all_tools": True, "verifier_mode": "none"},
    "A4v13_no_retrieval": {"method": "A4v13_no_retrieval", "kind": "trace", "disable_bleaching_tools": True, "disable_retrieval": True, "verifier_mode": "grok"},
    "A6v13_no_prior": {"method": "A6v13_no_prior", "kind": "trace", "disable_bleaching_tools": True, "use_baseline_prior": False, "verifier_mode": "grok"},
    "A7v13_plus_bleaching_tools": {"method": "A7v13_plus_bleaching_tools", "kind": "trace", "verifier_mode": "grok"},
    "A3v13_no_scope": {"method": "A3v13_no_scope", "kind": "trace", "disable_bleaching_tools": True, "verifier_mode": "grok",
                       "_region_override": "whole_image", "_ns": "noscope"},
    "A3v13_no_genus_list": {"method": "A3v13_no_genus_list", "kind": "trace", "disable_bleaching_tools": True, "verifier_mode": "grok",
                            "_genus_list": False, "_ns": "nogenuslist"},
}


def trace(row, method: str) -> Dict[str, Any]:
    cfg = dict(V13_SPECS[method])
    with runtime.variant(ns=cfg.get("_ns", "base"), region_override=cfg.get("_region_override"),
                         genus_list=cfg.get("_genus_list", True)):
        rec = core.run_trace_variant_with_verifier(row, cfg)
    rec = dict(rec)
    rec["split"] = row["split"]
    rec["v13_pre_guard_answer"] = _pre_guard_answer(rec)
    return rec


def _pre_guard_answer(rec: Dict[str, Any]) -> Optional[str]:
    for step in rec.get("trajectory") or []:
        d = step.get("decision") or {}
        if d.get("action") == "answer" and isinstance(step.get("final_response"), dict):
            return core.extract_answer_field(step["final_response"].get("answer", ""))
    return None


def trace_no_guard_from(rec: Dict[str, Any]) -> Dict[str, Any]:
    """A3 without GUARD = the same trajectory's pre-verification answer (no extra calls)."""
    out = {k: rec.get(k) for k in ["sample_id", "image_id", "split", "question", "answer_gt", "question_type", "gt_binary",
                                     "tool_sequence", "n_tool_calls", "intent", "plan"]}
    pre = None
    for step in rec.get("trajectory") or []:
        if (step.get("decision") or {}).get("action") == "answer":
            pre = step.get("final_response") or {}
            break
    pre = pre or {}
    out.update({"method": "A5v13_no_guard", "answer_pred": core.extract_answer_field(pre.get("answer", "")),
                "health_status_pred": pre.get("health_status"), "finalizer_source": pre.get("finalizer_source"),
                "prediction": pre, "derived_from": rec.get("method"), "error": rec.get("error")})
    return out


# =========================== GUARD + one repair (shared) ===========================
def guard_and_repair(row, agent_record: Dict[str, Any], verifier_mode: str = "grok") -> Dict[str, Any]:
    """Same verification and single repair round as core.run_trace_variant_with_verifier."""
    verifier = core.verify_agent_answer_variant(row, agent_record, verifier_mode)
    repair_rounds, used = [], False
    if verifier is not None:
        for i in range(core.MAX_REPAIR_ROUNDS):
            if core.verifier_says_supported(verifier) or not core.V12_VERIFIER_REPAIR:
                break
            repaired = core.repair_final_with_verifier(row, agent_record, verifier)
            rec2 = dict(agent_record)
            rec2["final"] = repaired
            rec2["trajectory"] = list(agent_record.get("trajectory", [])) + [
                {"step": "guard_repair", "decision": {"action": "repair_answer"}, "verifier_before": verifier,
                 "final_response": repaired}]
            repair_rounds.append({"repair_round": i + 1, "verifier_before": verifier, "repaired_final": repaired})
            agent_record, used = rec2, True
            verifier = core.verify_agent_answer_variant(row, agent_record, verifier_mode)
    agent_record["verifier"] = verifier
    agent_record["repair_rounds"] = repair_rounds
    agent_record["verifier_repair_used"] = used
    return agent_record


# =========================== E5: same-information single call ======================
def single_call(row, method: str = "E5_single_call_same_info") -> Dict[str, Any]:
    focus = runtime.core.focus_metadata(row)
    hits = core.retrieve_knowledge(row["question"], genus=None, top_k=5)
    knowledge = "\n".join(f"- {h['text']}" for h in hits)
    images = [row["local_image_path"]] + ([focus["focus_image_path"]] if focus["uses_crop"] else [])
    prompt = f"""
Answer the exact CoralVQA question.
Image 1 is the full image.{' Image 2 is the question-focused crop (' + focus['region_label'] + ').' if focus['uses_crop'] else ''}
Question: {row['question']}
Question type: {row['question_type']}

Relevant coral knowledge:
{knowledge}

Rules:{core._tool_answer_rules(row)}

Return JSON with keys:
- answer: shortest direct answer
- confidence: number from 0 to 1
- evidence: one short phrase grounded in the image
"""
    spec = gateway.make_spec("openai", config.GPT5, prompt, core.JSON_ONLY, images, expect_json=True,
                             max_tokens=config.GPT5_V13_MIN_OUTPUT, purpose="single_call", effort=config.GPT5_V13_EFFORT)
    r = gateway.as_v12_result(spec, gateway.lookup(spec))
    pred = r["json"] if isinstance(r.get("json"), dict) else {}
    ans = core.canonicalize_for_intent(row, pred.get("answer", r.get("text", "")))
    rec = _baseline_record(row, method, ans, pred, None, pred.get("confidence"), r)
    rec["n_images"] = len(images)
    return rec


# =========================== E6: matched-budget majority vote ======================
def self_consistency(row, k: int, method: str = "E6_self_consistency") -> Dict[str, Any]:
    answers, keys, missing = [], [], []
    for i in range(k):
        try:
            r = strict_gpt5(row, "tmp", sample_idx=i)
            answers.append(r["answer_pred"]); keys += r["request_keys"]
        except gateway.NeedCall as e:
            missing.append(str(e))
    if missing:
        raise gateway.NeedCall(",".join(missing))
    normed = [core.norm_vote_answer(a, row["question"], row["question_type"]) for a in answers]
    counts: Dict[str, int] = {}
    for a in normed:
        if a and not core.answer_is_unclear(a):
            counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=lambda a: (counts[a], -normed.index(a))) if counts else (normed[0] if normed else "")
    rec = {"sample_id": row["sample_id"], "image_id": row["image_id"], "split": row["split"], "question": row["question"],
           "answer_gt": row["answer_gt"], "question_type": row["question_type"], "gt_binary": row.get("gt_binary"),
           "method": method, "answer_pred": best, "samples": answers, "k": k, "request_keys": keys, "error": None,
           "health_status_pred": None}
    return rec


# =========================== E7a: generic tool-using agent =========================
AGENT_MAX_TOOL_CALLS = 6
AGENT_MAX_TURNS = 8


def _agent_tools() -> List[Dict[str, Any]]:
    region = {"type": "string", "enum": runtime.REGIONS,
              "description": "Image region to inspect; whole_image for the full image."}
    D = core.TOOL_DESCRIPTIONS
    return [
        {"type": "function", "name": "baseline_prior", "description": D["baseline_prior"],
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "focused_vqa", "description": D["focused_vqa"],
         "parameters": {"type": "object", "properties": {"region": region}, "required": ["region"], "additionalProperties": False}},
        {"type": "function", "name": "genus_identification", "description": D["genus_identification"],
         "parameters": {"type": "object", "properties": {"region": region}, "required": ["region"], "additionalProperties": False}},
        {"type": "function", "name": "image_summary", "description": D["image_summary"],
         "parameters": {"type": "object", "properties": {"region": region}, "required": ["region"], "additionalProperties": False}},
        {"type": "function", "name": "retrieval", "description": D["retrieval"],
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "genus": {"type": "string"}},
                        "required": ["query"], "additionalProperties": False}},
    ]


AGENT_SYSTEM_GENERIC = """You are a general-purpose visual question answering agent for coral reef images.
You may call the provided tools in any order, inspect their outputs, and decide what to do next.
You may make at most {max_calls} tool calls in total. Tool outputs are evidence, not ground truth.
When you are ready, reply with only JSON: {{"answer": "<shortest direct answer>", "confidence": <0-1>, "evidence": ["..."]}}."""


def _run_agent_tool(row, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "baseline_prior":
        with runtime.variant(ns="base"):
            return core.tool_baseline_prior(row)
    if name == "retrieval":
        return core.tool_retrieval(row, query=args.get("query") or row["question"], genus=args.get("genus"))
    region = args.get("region") if args.get("region") in runtime.REGIONS else "whole_image"
    with runtime.variant(ns=f"agent_{region}", region_override=region):
        if name == "focused_vqa":
            return core.tool_focused_vqa(row)
        if name == "genus_identification":
            return core.tool_genus_identification(row)
        if name == "image_summary":
            return core.tool_image_summary(row)
    return {"error": f"unknown tool {name}"}


def generic_agent(row, method: str = "E7a_generic_gpt5_agent") -> Dict[str, Any]:
    instructions = AGENT_SYSTEM_GENERIC.format(max_calls=AGENT_MAX_TOOL_CALLS)
    user_text = (f"Question: {row['question']}\nQuestion type: {row['question_type']}\n"
                 f"Answer-format rules:{core._tool_answer_rules(row)}")
    base_input = [{"role": "system", "content": instructions},
                  {"role": "user", "content": [{"type": "input_text", "text": user_text},
                                               {"type": "input_image", "image_path": str(row["local_image_path"])}]}]
    history: List[Dict[str, Any]] = []
    observations, keys, calls_used, final = [], [], 0, None
    for turn in range(AGENT_MAX_TURNS):
        allow_tools = calls_used < AGENT_MAX_TOOL_CALLS
        body = {"model": config.GPT5, "input": base_input + history, "tools": _agent_tools(),
                "tool_choice": "auto" if allow_tools else "none", "max_output_tokens": config.GPT5_V13_MIN_OUTPUT,
                "reasoning": {"effort": config.GPT5_V13_EFFORT}, "store": False}
        spec = gateway.make_spec("openai", config.GPT5, "", "", [], max_tokens=config.GPT5_V13_MIN_OUTPUT,
                                 purpose="agent_generic", raw_body=body, meta={"turn": turn})
        resp = gateway.lookup(spec)
        keys.append(resp.get("key"))
        text = resp.get("text") or ""
        calls = []
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "output" in obj:
                calls = [it for it in obj["output"] if it.get("type") == "function_call"]
                text = obj.get("text") or ""
        except Exception:
            pass
        if calls and allow_tools:
            missing = []
            for fc in calls[: AGENT_MAX_TOOL_CALLS - calls_used]:
                try:
                    args = json.loads(fc.get("arguments") or "{}")
                except Exception:
                    args = {}
                try:
                    result = _run_agent_tool(row, fc.get("name"), args)
                except gateway.NeedCall as e:
                    missing.append(str(e)); continue
                history.append({"type": "function_call", "call_id": fc.get("call_id"), "name": fc.get("name"),
                                "arguments": fc.get("arguments") or "{}"})
                history.append({"type": "function_call_output", "call_id": fc.get("call_id"),
                                "output": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
                observations.append({"step": len(observations) + 1, "tool": fc.get("name"), "args": args, "result": result})
            if missing:
                raise gateway.NeedCall(",".join(missing))
            calls_used = len(observations)
            continue
        final = core.parse_json_loose(text) if text else {}
        break
    final = final if isinstance(final, dict) else {}
    answer = core.canonicalize_for_intent(row, final.get("answer", ""))
    final_dict = {"answer": answer, "confidence": final.get("confidence"), "evidence": final.get("evidence"),
                  "health_status": "not_applicable", "finalizer_source": "generic_agent"}
    agent_record = {"final": final_dict, "observations": observations, "trajectory": [], "intent": core.question_intent(row)}
    pre_guard = answer
    with runtime.variant(ns="base"):
        agent_record = guard_and_repair(row, agent_record, "grok")
    fin = agent_record["final"]
    return {"sample_id": row["sample_id"], "image_id": row["image_id"], "split": row["split"], "question": row["question"],
            "answer_gt": row["answer_gt"], "question_type": row["question_type"], "gt_binary": row.get("gt_binary"),
            "method": method, "answer_pred": core.extract_answer_field(fin.get("answer", "")), "v13_pre_guard_answer": pre_guard,
            "health_status_pred": fin.get("health_status"), "prediction": fin, "observations": observations,
            "tool_sequence": [o["tool"] for o in observations], "n_tool_calls": len(observations),
            "verifier": agent_record.get("verifier"), "verifier_repair_used": agent_record.get("verifier_repair_used"),
            "repair_rounds": agent_record.get("repair_rounds"), "agent_turn_keys": keys, "error": None}


# =========================== E8: blinded evidence-quality judge ===================
JUDGE_SYSTEM_V13 = """You are an independent reviewer of evidence quality for coral reef visual question answering.
You see the image, the question, a system's final answer and the evidence trace it produced. You do not know which
system produced it. Rate only the evidence, not whether the answer matches any hidden reference.
Return only JSON with keys:
- grounded: 0, 1 or 2 (is the evidence visible in the image?)
- relevant: 0, 1 or 2 (does the evidence address the exact question?)
- sufficient: 0, 1 or 2 (does the evidence justify the final answer?)
- contradicts_answer: true/false (does any evidence contradict the final answer?)
- overall: integer 1-5
- note: one short sentence"""


def judge(row, system_record: Dict[str, Any], method: str, provider: str = "together",
          model: str = config.GEMMA) -> Dict[str, Any]:
    obs = system_record.get("observations") or []
    trace_txt = core.observation_for_prompt(obs, max_chars=5000) if obs else "(no tool evidence)"
    prompt = (f"Question: {row['question']}\nFinal answer: {system_record.get('answer_pred')}\n\n"
              f"Evidence trace:\n{trace_txt}")
    spec = gateway.make_spec(provider, model, prompt, JUDGE_SYSTEM_V13, [row["local_image_path"]],
                             expect_json=True, max_tokens=600, purpose="judge_evidence")
    r = gateway.as_v12_result(spec, gateway.lookup(spec))
    return {"sample_id": row["sample_id"], "split": row["split"], "method": method,
            "judged_method": system_record.get("method"), "rating": r.get("json"), "error": r.get("error"),
            "request_keys": [r.get("request_key")]}
