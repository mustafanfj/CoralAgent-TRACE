"""SQLite store for requests, responses, batches, per-row results and the spend ledger.

One file (outputs/v13/llm_store.sqlite) so a crash or restart never loses state and
never causes a batch to be submitted twice.
"""
import json
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  key TEXT PRIMARY KEY, provider TEXT, model TEXT, stage TEXT, spec TEXT, created REAL);
CREATE TABLE IF NOT EXISTS responses (
  key TEXT PRIMARY KEY, provider TEXT, status TEXT, attempts INTEGER, text TEXT, usage TEXT,
  cost REAL, model_returned TEXT, batch_id TEXT, raw TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY, provider TEXT, stage TEXT, kind TEXT, n INTEGER, keys TEXT,
  status TEXT, worst REAL, actual REAL, created REAL, finished REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS results (
  method TEXT, sample_id TEXT, record TEXT, updated REAL, PRIMARY KEY (method, sample_id));
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path=None):
        config.ensure_dirs()
        self.path = str(path or config.STORE_DB)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---- requests / responses ----
    def add_request(self, key: str, spec: Dict[str, Any], stage: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,?)",
            (key, spec["provider"], spec["model"], stage, json.dumps(spec, ensure_ascii=False), time.time()))
        self.db.commit()

    def get_spec(self, key: str) -> Optional[Dict[str, Any]]:
        r = self.db.execute("SELECT spec FROM requests WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def get_response(self, key: str) -> Optional[Dict[str, Any]]:
        r = self.db.execute(
            "SELECT status, attempts, text, usage, cost, model_returned, batch_id FROM responses WHERE key=?",
            (key,)).fetchone()
        if not r:
            return None
        return {"status": r[0], "attempts": r[1], "text": r[2], "usage": json.loads(r[3] or "null"),
                "cost": r[4], "model_returned": r[5], "batch_id": r[6]}

    def put_response(self, key: str, provider: str, status: str, text: str, usage: Any, cost: float,
                     model_returned: str, batch_id: str, raw: Any) -> None:
        prev = self.get_response(key)
        attempts = (prev["attempts"] if prev else 0) + 1
        self.db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (key, provider, status, attempts, text, json.dumps(usage, default=str), float(cost or 0.0),
             model_returned, batch_id, json.dumps(raw, default=str)[:200000], time.time()))
        self.db.commit()

    # ---- batches ----
    def add_batch(self, batch_id: str, provider: str, stage: str, kind: str, keys: List[str],
                  worst: float, meta: Dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (batch_id, provider, stage, kind, len(keys), json.dumps(keys), "submitted", worst, None,
             time.time(), None, json.dumps({**meta, "_created": time.time()}, default=str)))
        self.db.commit()

    def open_batches(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT batch_id, provider, stage, kind, keys, worst, meta FROM batches WHERE status='submitted'"
        rows = self.db.execute(q).fetchall()
        out = [{"batch_id": r[0], "provider": r[1], "stage": r[2], "kind": r[3], "keys": json.loads(r[4]),
                "worst": r[5], "meta": json.loads(r[6] or "{}")} for r in rows]
        return [b for b in out if provider is None or b["provider"] == provider]

    def close_batch(self, batch_id: str, status: str, actual: float) -> None:
        self.db.execute("UPDATE batches SET status=?, actual=?, finished=? WHERE batch_id=?",
                        (status, actual, time.time(), batch_id))
        self.db.commit()

    def batches_for_stage(self, stage: str) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT batch_id, provider, kind, n, status, worst, actual, created, finished FROM batches WHERE stage=?",
            (stage,)).fetchall()
        keys = ["batch_id", "provider", "kind", "n", "status", "worst", "actual", "created", "finished"]
        return [dict(zip(keys, r)) for r in rows]

    # ---- ledger ----
    def spent(self, provider: str) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(actual),0) FROM batches WHERE provider=? AND actual IS NOT NULL",
                            (provider,)).fetchone()
        return float(r[0] or 0.0)

    def reserved(self, provider: str) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(worst),0) FROM batches WHERE provider=? AND status='submitted'",
                            (provider,)).fetchone()
        return float(r[0] or 0.0)

    # ---- results ----
    def put_result(self, method: str, sample_id: str, record: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                        (method, str(sample_id), json.dumps(record, ensure_ascii=False, default=str), time.time()))

    def has_result(self, method: str, sample_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM results WHERE method=? AND sample_id=?",
                               (method, str(sample_id))).fetchone() is not None

    def results(self, method: str) -> List[Dict[str, Any]]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT record FROM results WHERE method=?", (method,))]

    def result_counts(self) -> Dict[str, int]:
        return dict(self.db.execute("SELECT method, COUNT(*) FROM results GROUP BY method").fetchall())

    def commit(self) -> None:
        self.db.commit()

    # ---- kv ----
    def get_kv(self, k: str, default=None):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_kv(self, k: str, v) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, default=str)))
        self.db.commit()

    def usage_rows(self) -> Iterable:
        return self.db.execute(
            "SELECT r.key, q.stage, r.provider, q.model, r.status, r.attempts, r.usage, r.cost, r.model_returned "
            "FROM responses r JOIN requests q ON r.key=q.key")
