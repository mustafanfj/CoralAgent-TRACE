"""Offline mock of the batch adapters for testing the pipeline end to end (no network, no cost)."""
import json
import random
from typing import Any, Dict, List, Tuple


class MockAdapter:
    def __init__(self, provider: str, fail_rate: float = 0.0, seed: int = 0):
        self.provider = provider
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)
        self.batches: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.n = 0

    def submit(self, specs, name) -> Tuple[str, Dict[str, Any]]:
        self.n += 1
        bid = f"mock_{self.provider}_{self.n}"
        self.batches[bid] = dict(specs)
        return bid, {"n": len(specs)}

    def status(self, batch_id, meta) -> str:
        return "done"

    def fetch(self, batch_id, meta) -> List[Dict[str, Any]]:
        return [self._answer(k, s) for k, s in self.batches[batch_id].items()]

    def _answer(self, key, spec) -> Dict[str, Any]:
        if self.rng.random() < self.fail_rate:
            return {"key": key, "status": "ok", "text": "", "usage": {"input_tokens": 900, "output_tokens": 2000},
                    "cost_usd": None, "model": spec["model"], "raw": {}}
        p, prompt = spec["purpose"], spec.get("prompt") or ""
        if spec.get("raw_body") is not None:                       # generic agent turn
            n_prev = sum(1 for it in spec["raw_body"]["input"] if isinstance(it, dict) and it.get("type") == "function_call")
            if n_prev == 0:
                out = {"output": [{"type": "function_call", "call_id": "c1", "name": "focused_vqa",
                                   "arguments": json.dumps({"region": "whole_image"})}], "status": "completed", "text": ""}
                text = json.dumps(out)
            else:
                text = json.dumps({"answer": "Yes", "confidence": 0.6, "evidence": ["mock"]})
        elif p == "baseline_strict":
            text = self.rng.choice(["Yes", "No", "Acropora", "Brown", "2"])
        elif "independent verifier" in prompt:
            sup = self.rng.random() < 0.6
            text = json.dumps({"supported": sup, "support_level": "supported" if sup else "unsupported",
                               "exact_question_answered": True, "contradictions": [], "missing_evidence": [],
                               "suggested_answer": None if sup else "No", "repair_instruction": None,
                               "verifier_confidence": 0.7})
        elif "likely_genus" in prompt:
            text = json.dumps({"likely_genus": "Acropora", "direct_answer_candidate": "Acropora", "confidence": 0.7,
                               "candidates": [], "alternatives": [], "morphology_evidence": "branching", "caveat": ""})
        elif "evidence quality" in (spec.get("instructions") or ""):
            text = json.dumps({"grounded": 2, "relevant": 2, "sufficient": 1, "contradicts_answer": False,
                               "overall": 4, "note": "mock"})
        elif "scene_summary" in prompt:
            text = json.dumps({"direct_answer_candidate": "Branching", "scene_summary": "mock", "visible_coral_count": 2,
                               "dominant_morphology": "branching", "dominant_colors": ["brown"], "health_visual_notes": "",
                               "uncertainty": ""})
        elif "chart_present" in prompt:
            text = json.dumps({"chart_present": False, "confidence": 0.9, "chart_type": "none", "evidence": "",
                               "usable_for_color_calibration": False})
        else:
            text = json.dumps({"answer": self.rng.choice(["Yes", "No", "Brown", "Acropora", "2"]), "confidence": 0.7,
                               "visual_evidence": "mock", "uncertainty": "", "health_status": "unclear",
                               "evidence": ["mock"], "rationale": "mock"})
        usage = ({"input_tokens": 900, "output_tokens": 300} if self.provider == "openai" else
                 {"prompt_tokens": 1800, "completion_tokens": 80, "reasoning_tokens": 600})
        return {"key": key, "status": "ok", "text": text, "usage": usage, "cost_usd": None, "model": spec["model"], "raw": {}}
