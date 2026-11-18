"""Paths, models, prices and budget caps for the v13 revision runs.

Everything here is fixed by the pre-registered protocol (PROTOCOL.md). Budget caps can
be tightened (never raised) by control/CONTROL.json.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CORAL_ROOT", str(REPO / "workdir")))   # data + run state (see README)

DATA = ROOT / "data"
IMAGES = DATA / "images"                       # the 500 bank images, byte-identical to v12 inputs
CORALVQA_JSONL = DATA / "data" / "CoralVQA_train.jsonl"
BANK_CSV = DATA / "outputs" / "active_pilot_500images_v12_TRACE_bank500_q6.csv"
V12_BASELINES = DATA / "outputs" / "baselines"
V12_ABLATIONS = DATA / "outputs" / "ablations"
V12_TOOL_CACHE = DATA / "cache" / "tool_cache_v12_TRACE_bank500_q6.json"

RUN = ROOT / "outputs" / "v13"                 # all new run state (not in git)
STORE_DB = RUN / "llm_store.sqlite"
BATCH_DIR = RUN / "batch_files"
WORK = RUN / "work"                            # crops, masks, v12-core scratch

RESULTS = REPO / "results" / "v13"             # per-row results pushed to git (compressed)
REPORTS = REPO / "reports"
CONTROL = REPO / "control" / "CONTROL.json"

EXPERIMENT_TAG = "v13_fix"

# ---- Models (pinned) --------------------------------------------------------------
GPT5 = os.environ.get("CORAL_GPT5_MODEL", "gpt-5")
GPT41 = os.environ.get("CORAL_GPT41_MODEL", "gpt-4.1")
GROK = os.environ.get("CORAL_GROK_MODEL", "grok-4.3")
GEMMA = os.environ.get("CORAL_GEMMA_MODEL", "google/gemma-4-31B-it")

# v13 fix 1: output budget for GPT-5 tool / finalizer / repair / agent calls.
GPT5_V13_EFFORT = "low"
GPT5_V13_MIN_OUTPUT = 2000

# ---- Prices, USD per 1M tokens (list prices; batch discount applied separately) ----
PRICES = {
    GPT5: (1.25, 10.00),
    GPT41: (2.00, 8.00),
    GROK: (1.25, 2.50),
    GEMMA: (0.39, 0.97),
}
# Discount factor applied to list price for batch jobs. Only OpenAI's 50% is certain;
# the others are treated as full price for budget guarding (actual cost is logged).
BATCH_FACTOR = {"openai": 0.5, "xai": 1.0, "together": 1.0}

# Conservative image-token estimates per image for the worst-case budget check.
IMAGE_TOKENS_WORST = {"openai": 1105, "xai": 1800, "together": 2600}

# ---- Budget caps (USD). The OpenAI account holds $40; keep a $2 safety margin. -------
DEFAULT_CAPS = {"openai": 38.0, "xai": 80.0, "together": 10.0}

# ---- Safety rules -----------------------------------------------------------------
CANARY_SIZE = 10
STOP_FAILURE_RATE = 0.05
STOP_MIN_N = 50              # rounds smaller than this stop only with >= STOP_MIN_BAD failures
STOP_MIN_BAD = 10
MAX_ATTEMPTS = 2            # first submission + at most one resubmission of failed requests
MAX_OPENAI_FILE_BYTES = 180 * 1024 * 1024
MAX_TOGETHER_FILE_BYTES = 90 * 1024 * 1024
MAX_LINES_PER_FILE = 40000
POLL_SECONDS = 60
MAX_BATCH_WAIT_HOURS = 26     # OpenAI/xAI/Together batch windows are 24 h


def load_control() -> dict:
    if CONTROL.exists():
        return json.loads(CONTROL.read_text(encoding="utf-8"))
    return {"allowed_stages": [], "caps": {}}


def caps() -> dict:
    out = dict(DEFAULT_CAPS)
    for k, v in (load_control().get("caps") or {}).items():
        if k in out:
            out[k] = min(out[k], float(v)) if k == "openai" else float(v)
    return out


def ensure_dirs() -> None:
    for d in [RUN, BATCH_DIR, WORK, RESULTS, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)
