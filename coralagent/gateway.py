"""Batch-only model gateway ("replay until complete").

Every model call in the v12 pipeline goes through `call_llm`. Here that function never
contacts a provider. It builds a deterministic request spec, looks the response up in the
store, and either returns it or registers the request as pending and raises NeedCall.
The runner then submits all pending requests as provider batch jobs and replays the
pipeline. Deterministic code + stored responses = identical results on every replay.
"""
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from . import config

PENDING: Dict[str, Dict[str, Any]] = {}
STAGE = {"name": "unset"}
_IMAGE_SHA: Dict[str, str] = {}
STORE = None          # set by runner


class NeedCall(Exception):
    """A model response is not available yet; the request was registered as pending."""


class WaitDependency(Exception):
    """A non-model dependency (e.g. the strict GPT-5 prior for this row) is not available yet."""


def _image_sha(path: str) -> str:
    p = str(path)
    if p not in _IMAGE_SHA:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _IMAGE_SHA[p] = h.hexdigest()
    return _IMAGE_SHA[p]


def spec_key(spec: Dict[str, Any]) -> str:
    s = dict(spec)
    s["images"] = [_image_sha(p) for p in spec.get("images") or []]
    s.pop("meta", None)
    blob = json.dumps(s, sort_keys=True, ensure_ascii=False, default=str)
    return "k" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]


def make_spec(provider: str, model: str, prompt: str, instructions: str = "", images: Optional[List[str]] = None,
              expect_json: bool = False, max_tokens: int = 900, purpose: str = "generic", effort: Optional[str] = None,
              sample_idx: int = 0, raw_body: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    return {
        "provider": provider, "model": model, "purpose": purpose, "instructions": instructions or "",
        "prompt": prompt or "", "images": [str(p) for p in (images or []) if p],
        "expect_json": bool(expect_json), "max_tokens": int(max_tokens), "effort": effort,
        "sample_idx": int(sample_idx), "raw_body": raw_body, "meta": meta or {},
    }


def is_bad(spec: Dict[str, Any], status: str, text: str) -> bool:
    """A response counts as failed if the provider errored, the text is empty, or JSON was expected and
    could not be parsed. Failed responses get at most one resubmission (config.MAX_ATTEMPTS)."""
    if status != "ok":
        return True
    if spec.get("raw_body") is not None:      # agent turns: validity checked by the agent loop
        return False
    if not str(text or "").strip():
        return True
    if spec.get("expect_json"):
        from . import runtime
        parsed = runtime.core.parse_json_loose(text)
        if not isinstance(parsed, dict) or parsed.get("parse_error"):
            return True
    return False


def lookup(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stored response for this spec, or register it and raise NeedCall."""
    key = spec_key(spec)
    resp = STORE.get_response(key)
    if resp is not None:
        final = (not is_bad(spec, resp["status"], resp["text"])) or resp["attempts"] >= config.MAX_ATTEMPTS
        if final:
            resp["key"] = key
            return resp
    STORE.add_request(key, spec, STAGE["name"])
    PENDING[key] = spec
    raise NeedCall(key)


def as_v12_result(spec: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a stored response into the dict shape v12's call_llm returned."""
    from . import runtime
    text = resp.get("text") or ""
    error = None if resp.get("status") == "ok" and text.strip() else f"batch_status={resp.get('status')} empty_or_error"
    parsed = runtime.core.parse_json_loose(text) if spec.get("expect_json") else None
    return {"text": text, "json": parsed, "provider": spec["provider"], "model": spec["model"],
            "model_returned": resp.get("model_returned"), "latency_s": None, "usage": resp.get("usage"),
            "error": error, "request_key": resp.get("key"), "attempts": resp.get("attempts")}


# ---- token / cost estimates for the budget guard ----------------------------------
def _openai_image_tokens(path: str) -> int:
    try:
        with Image.open(path) as im:
            w, h = im.size
        s = min(1.0, 1280 / max(w, h))
        w, h = w * s, h * s
        s2 = min(1.0, 768 / min(w, h))
        w, h = w * s2, h * s2
        return 85 + 170 * math.ceil(w / 512) * math.ceil(h / 512)
    except Exception:
        return config.IMAGE_TOKENS_WORST["openai"]


def estimate_tokens(spec: Dict[str, Any]) -> Dict[str, int]:
    text = (spec.get("instructions") or "") + (spec.get("prompt") or "")
    if spec.get("raw_body") is not None:
        text = json.dumps(spec["raw_body"].get("input", ""), default=str)
        text = text.split("data:image")[0] + "".join(t[-2000:] for t in text.split("data:image")[1:])
    tin = int(len(text) / 3.0) + 50
    for p in spec.get("images") or []:
        tin += _openai_image_tokens(p) if spec["provider"] == "openai" else config.IMAGE_TOKENS_WORST[spec["provider"]]
    if spec.get("raw_body") is not None:
        tin += config.IMAGE_TOKENS_WORST["openai"] * str(spec["raw_body"]).count("input_image")
    return {"in": tin, "out_worst": int(spec.get("max_tokens") or 0)}


def worst_cost(spec: Dict[str, Any]) -> float:
    t = estimate_tokens(spec)
    if spec["provider"] == "xai":       # Grok reasoning tokens are billed on top of max_tokens
        t["out_worst"] += 2500
    pin, pout = config.PRICES.get(spec["model"], (5.0, 15.0))
    fac = config.BATCH_FACTOR.get(spec["provider"], 1.0)
    return (t["in"] * 1.3 * pin + t["out_worst"] * pout) / 1e6 * fac


def actual_cost(provider: str, model: str, usage: Any, reported_usd: Optional[float] = None) -> float:
    if reported_usd is not None and reported_usd > 0:
        return float(reported_usd)
    if not isinstance(usage, dict):
        return 0.0
    tin = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    tout = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    rdet = usage.get("completion_tokens_details") or {}
    if provider == "xai" and isinstance(rdet, dict) and "output_tokens" not in usage:
        tout += rdet.get("reasoning_tokens", 0) or usage.get("reasoning_tokens", 0) or 0
    pin, pout = config.PRICES.get(model, (5.0, 15.0))
    return (tin * pin + tout * pout) / 1e6 * config.BATCH_FACTOR.get(provider, 1.0)
