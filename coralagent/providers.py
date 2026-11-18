"""Batch adapters for OpenAI, xAI and Together. Batch APIs only: no synchronous inference.

Each adapter: submit(specs_by_key, name) -> (batch_id, meta); status(batch_id, meta) -> 'running' |
'done' | 'failed'; fetch(batch_id, meta) -> list of dicts {key, status, text, usage, cost_usd, model}.
Free metadata calls (file upload, batch create/poll/download) are the only other API traffic.
"""
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import config


def _data_url(path: str) -> str:
    from . import runtime
    return runtime.core.image_to_data_url(path)


def _resolve_raw_images(obj):
    """Agent turns keep images as {'type':'input_image','image_path':p}; expand to data URLs at submit time."""
    if isinstance(obj, dict):
        if obj.get("type") == "input_image" and "image_path" in obj:
            return {"type": "input_image", "image_url": _data_url(obj["image_path"])}
        return {k: _resolve_raw_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_raw_images(v) for v in obj]
    return obj


# =============================== OpenAI ==========================================
class OpenAIBatch:
    provider = "openai"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @staticmethod
    def body(spec: Dict[str, Any]) -> Dict[str, Any]:
        if spec.get("raw_body") is not None:
            return _resolve_raw_images(spec["raw_body"])
        content = [{"type": "input_text", "text": spec["prompt"]}]
        for p in spec.get("images") or []:
            content.append({"type": "input_image", "image_url": _data_url(p)})
        inp = []
        if spec.get("instructions"):
            inp.append({"role": "system", "content": spec["instructions"]})
        inp.append({"role": "user", "content": content})
        body = {"model": spec["model"], "input": inp, "max_output_tokens": spec["max_tokens"]}
        if spec.get("effort"):
            body["reasoning"] = {"effort": spec["effort"]}
        return body

    def submit(self, specs: Dict[str, Dict[str, Any]], name: str) -> Tuple[str, Dict[str, Any]]:
        # OpenAI rejects any line whose model differs from the rest of its batch file
        # ("mismatched_model"), so each file holds requests for one model only.
        buf, size, n, cur_model = io.StringIO(), 0, 0, None
        parts = []
        for key, spec in sorted(specs.items(), key=lambda kv: kv[1]["model"]):
            body = self.body(spec)
            line = json.dumps({"custom_id": key, "method": "POST", "url": "/v1/responses", "body": body},
                              ensure_ascii=False) + "\n"
            b = len(line.encode("utf-8"))
            if n and (body.get("model") != cur_model or size + b > config.MAX_OPENAI_FILE_BYTES
                      or n >= config.MAX_LINES_PER_FILE):
                parts.append(buf.getvalue()); buf, size, n = io.StringIO(), 0, 0
            cur_model = body.get("model")
            buf.write(line); size += b; n += 1
        if n:
            parts.append(buf.getvalue())
        batch_ids = []
        for i, text in enumerate(parts):
            path = config.BATCH_DIR / f"{name}_openai_{i}.jsonl"
            path.write_text(text, encoding="utf-8")
            with open(path, "rb") as f:
                fobj = self.client.files.create(file=f, purpose="batch")
            b = self.client.batches.create(input_file_id=fobj.id, endpoint="/v1/responses",
                                           completion_window="24h", metadata={"name": name[:60]})
            batch_ids.append(b.id)
        return batch_ids[0] if len(batch_ids) == 1 else "multi:" + ",".join(batch_ids), {"ids": batch_ids}

    def batch_error(self, batch_id: str, meta: Dict[str, Any]) -> str:
        return "\n".join(f"{i}: {self.client.batches.retrieve(i).errors}" for i in meta["ids"])

    def status(self, batch_id: str, meta: Dict[str, Any]) -> str:
        bs = [self.client.batches.retrieve(i) for i in meta["ids"]]
        states = [b.status for b in bs]
        for b in bs:
            if b.status in {"failed", "expired", "cancelled"}:
                print(f"OpenAI batch {b.id} {b.status}: {getattr(b, 'errors', None)}", flush=True)
        if all(s == "completed" for s in states):
            return "done"
        if any(s in {"failed", "expired", "cancelled"} for s in states) and \
                all(s in {"completed", "failed", "expired", "cancelled"} for s in states):
            return "failed_partial"
        return "running"

    def fetch(self, batch_id: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for i in meta["ids"]:
            b = self.client.batches.retrieve(i)
            for fid in [b.output_file_id, b.error_file_id]:
                if not fid:
                    continue
                text = self.client.files.content(fid).text
                for line in text.splitlines():
                    if line.strip():
                        out.append(self.parse_line(json.loads(line)))
        return out

    @staticmethod
    def parse_line(rec: Dict[str, Any]) -> Dict[str, Any]:
        key = rec.get("custom_id")
        resp = rec.get("response") or {}
        body = resp.get("body") or {}
        if rec.get("error") or resp.get("status_code", 200) != 200 or not body:
            return {"key": key, "status": "error", "text": "", "usage": body.get("usage"), "cost_usd": None,
                    "model": body.get("model"), "raw": rec}
        chunks = []
        for item in body.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") == "output_text" and c.get("text"):
                    chunks.append(c["text"])
        text = "\n".join(chunks)
        if any(item.get("type") == "function_call" for item in body.get("output") or []):
            text = json.dumps({"output": body.get("output"), "status": body.get("status"), "text": text})
        return {"key": key, "status": "ok", "text": text, "usage": body.get("usage"), "cost_usd": None,
                "model": body.get("model"), "raw": {"status": body.get("status"),
                                                    "incomplete": body.get("incomplete_details")}}


# =============================== xAI =============================================
class XAIBatch:
    provider = "xai"
    MAX_ADD_BYTES = 14 * 1024 * 1024

    def __init__(self):
        from xai_sdk import Client
        self.client = Client(api_key=os.environ["XAI_API_KEY"], timeout=3600)

    def submit(self, specs: Dict[str, Dict[str, Any]], name: str) -> Tuple[str, Dict[str, Any]]:
        from xai_sdk.chat import image, system, user
        b = self.client.batch.create(batch_name=name[:120])
        chunk, size = [], 0
        for key, spec in specs.items():
            parts = [spec["prompt"]] + [image(_data_url(p), detail="auto") for p in spec.get("images") or []]
            msgs = ([system(spec["instructions"])] if spec.get("instructions") else []) + [user(*parts)]
            chat = self.client.chat.create(model=spec["model"], max_tokens=spec["max_tokens"],
                                           batch_request_id=key, messages=msgs)
            b_est = sum(len(_data_url(p)) for p in spec.get("images") or []) + len(spec["prompt"]) + 4096
            if chunk and size + b_est > self.MAX_ADD_BYTES:
                self.client.batch.add(batch_id=b.batch_id, batch_requests=chunk)
                chunk, size = [], 0
                time.sleep(0.2)
            chunk.append(chat); size += b_est
        if chunk:
            self.client.batch.add(batch_id=b.batch_id, batch_requests=chunk)
        return b.batch_id, {"n": len(specs)}

    def status(self, batch_id: str, meta: Dict[str, Any]) -> str:
        b = self.client.batch.get(batch_id)
        st = b.state
        if st.num_requests >= meta.get("n", 0) and st.num_pending == 0:
            return "done"
        return "running"

    def fetch(self, batch_id: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        out, token = [], None
        while True:
            page = self.client.batch.list_batch_results(batch_id, limit=100, pagination_token=token)
            for r in page.results:
                key = r.batch_request_id
                if r.has_error:
                    out.append({"key": key, "status": "error", "text": "", "usage": None, "cost_usd": None,
                                "model": None, "raw": {"error": str(r.proto.error)}})
                    continue
                resp = r.response
                u = resp.usage
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0),
                         "completion_tokens": getattr(u, "completion_tokens", 0),
                         "reasoning_tokens": getattr(u, "reasoning_tokens", 0)}
                cost = None
                try:
                    cost = resp.cost_usd
                except Exception:
                    pass
                model = None
                try:
                    model = r.proto.response.completion_response.model
                except Exception:
                    pass
                out.append({"key": key, "status": "ok", "text": resp.content or "", "usage": usage,
                            "cost_usd": cost, "model": model, "raw": {}})
            token = page.pagination_token
            if not token:
                break
        return out


# =============================== Together ========================================
class TogetherBatch:
    provider = "together"

    def __init__(self):
        from together import Together
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    @staticmethod
    def body(spec: Dict[str, Any]) -> Dict[str, Any]:
        content = [{"type": "text", "text": spec["prompt"]}]
        for p in spec.get("images") or []:
            content.append({"type": "image_url", "image_url": {"url": _data_url(p)}})
        msgs = ([{"role": "system", "content": spec["instructions"]}] if spec.get("instructions") else []) + \
               [{"role": "user", "content": content}]
        return {"model": spec["model"], "messages": msgs, "max_tokens": spec["max_tokens"]}

    def submit(self, specs: Dict[str, Dict[str, Any]], name: str) -> Tuple[str, Dict[str, Any]]:
        parts, buf, size = [], [], 0
        for key, spec in specs.items():
            line = json.dumps({"custom_id": key, "body": self.body(spec)}, ensure_ascii=False) + "\n"
            b = len(line.encode("utf-8"))
            if buf and size + b > config.MAX_TOGETHER_FILE_BYTES:
                parts.append(buf); buf, size = [], 0
            buf.append(line); size += b
        if buf:
            parts.append(buf)
        ids = []
        for i, lines in enumerate(parts):
            path = config.BATCH_DIR / f"{name}_together_{i}.jsonl"
            path.write_text("".join(lines), encoding="utf-8")
            fobj = self.client.files.upload(file=str(path), purpose="batch-api")
            b = self.client.batches.create(endpoint="/v1/chat/completions", input_file_id=fobj.id)
            ids.append(getattr(b, "id", None) or getattr(getattr(b, "job", None), "id", None))
        return (ids[0] if len(ids) == 1 else "multi:" + ",".join(ids)), {"ids": ids, "n": len(specs)}

    def batch_error(self, batch_id: str, meta: Dict[str, Any]) -> str:
        out = []
        for i in meta["ids"]:
            b = self.client.batches.retrieve(i)
            out.append(f"{i}: status={b.status} error={getattr(b, 'error', None)} progress={getattr(b, 'progress', None)}")
            fid = getattr(b, "error_file_id", None)
            if fid:
                try:
                    raw = self.client.files.content(fid)
                    txt = raw.text if hasattr(raw, "text") else raw.read().decode("utf-8")
                    out.append("error file (first 1500 chars): " + txt[:1500])
                except Exception as e:
                    out.append(f"error file unreadable: {e!r}")
        return "\n".join(out)

    def status(self, batch_id: str, meta: Dict[str, Any]) -> str:
        states = [str(self.client.batches.retrieve(i).status or "").upper() for i in meta["ids"]]
        if all(s == "COMPLETED" for s in states):
            return "done"
        if all(s in {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"} for s in states):
            return "failed_partial"
        return "running"

    def fetch(self, batch_id: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for bid in meta["ids"]:
            b = self.client.batches.retrieve(bid)
            for fid in [getattr(b, "output_file_id", None), getattr(b, "error_file_id", None)]:
                if not fid:
                    continue
                raw = self.client.files.content(fid)
                text = raw.text if hasattr(raw, "text") else raw.read().decode("utf-8")
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = rec.get("custom_id")
                    body = (rec.get("response") or {}).get("body") or rec.get("body") or rec.get("response") or {}
                    try:
                        txt = body["choices"][0]["message"]["content"] or ""
                        out.append({"key": key, "status": "ok", "text": txt, "usage": body.get("usage"),
                                    "cost_usd": None, "model": body.get("model"), "raw": {}})
                    except Exception:
                        out.append({"key": key, "status": "error", "text": "", "usage": None, "cost_usd": None,
                                    "model": None, "raw": rec})
        return out


ADAPTERS = {"openai": OpenAIBatch, "xai": XAIBatch, "together": TogetherBatch}
_INSTANCES: Dict[str, Any] = {}


def adapter(provider: str):
    if provider not in _INSTANCES:
        _INSTANCES[provider] = ADAPTERS[provider]()
    return _INSTANCES[provider]
