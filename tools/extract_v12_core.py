"""Generate coralagent/v12_core.py from the frozen v12 notebook.

The v12 notebook mixes definitions with execution (downloads, API runs, plots).
This script keeps only what the v13 runner needs to reuse the v12 logic unchanged:
imports, function/class definitions, and module-level constants. Runtime state
(datasets, caches, clients, knowledge-base index, genus list) is created by
coralagent/runtime.py instead.

Usage:  python tools/extract_v12_core.py CoralAgent_TRACE.ipynb coralagent/v12_core.py
"""
import ast
import hashlib
import json
import sys

# Code cells that hold reusable definitions (indices in the v12 notebook).
CELLS = [9, 12, 16, 26, 28, 30, 32, 35, 37, 39, 42, 44]

# Upper-case names that are runtime state, not constants: created by runtime.py.
DENY = {
    "TOOL_CACHE", "KNOWN_GENUS_LIST", "ENVIRONMENT_SNAPSHOT",
}
# Lower-case module-level assignments that are needed (knowledge-base index).
ALLOW_LOWER = {"kb_df", "vectorizer", "kb_matrix"}

HEADER = '''"""AUTO-GENERATED from the v12 notebook by tools/extract_v12_core.py. Do not edit.

Source notebook sha256: {sha}
Kept: imports, function/class definitions and module-level constants of cells {cells}.
"""
import os
import re
import io
import gc
import sys
import ast
import json
import time
import math
import base64
import zipfile
import random
import shutil
import textwrap
import traceback
import importlib
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
try:
    from tqdm.std import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


def display(*args, **kwargs):  # notebook stub
    return None


def Markdown(x):  # notebook stub
    return x


# ---- Paths (set by environment; runtime.py fills these before import) ----
WORK_DIR = Path(os.environ.get("CORALAGENT_WORK_DIR", ".")).resolve()
DATA_DIR = WORK_DIR / "data"
IMAGE_DIR = WORK_DIR / "images"
SAMPLE_IMAGE_DIR = WORK_DIR / "sample_images"
OUTPUT_DIR = WORK_DIR / "outputs_v12_core"
FIG_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = WORK_DIR / "cache_v12_core"
for _d in [DATA_DIR, IMAGE_DIR, SAMPLE_IMAGE_DIR, OUTPUT_DIR, FIG_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# No live API clients: every model call goes through coralagent.gateway (batch only).
OPENAI_API_KEY = XAI_API_KEY = TOGETHER_API_KEY = None
openai_client = xai_client = together_client = None
KNOWN_GENUS_LIST: List[str] = []
TOOL_CACHE: Dict[str, Any] = {{}}
'''


def keep_assign(node) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = []
    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            names.append(t.value.id)
        else:
            return False
    for n in names:
        if n in DENY:
            return False
        if n in ALLOW_LOWER:
            continue
        if not (n.isupper() or (n.replace("_", "").isupper())):
            return False
    return True


def main(nb_path: str, out_path: str) -> None:
    raw = open(nb_path, "rb").read()
    nb = json.loads(raw)
    parts = [HEADER.format(sha=hashlib.sha256(raw).hexdigest(), cells=CELLS)]
    for i in CELLS:
        cell = nb["cells"][i]
        assert cell["cell_type"] == "code", i
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
        tree = ast.parse(src)
        kept = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kept.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                if any(x in (mod or "") for x in ["IPython", "huggingface_hub", "openai", "getpass"]) or \
                   any(n.split(".")[0] in {"IPython", "huggingface_hub", "openai", "getpass", "matplotlib"} for n in names):
                    continue
                kept.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and keep_assign(node):
                kept.append(node)
        parts.append(f"\n# ======================= v12 notebook cell {i} =======================\n")
        parts.append(ast.unparse(ast.Module(body=kept, type_ignores=[])))
        parts.append("\n")
    code = "\n".join(parts)
    compile(code, out_path, "exec")
    open(out_path, "w", encoding="utf-8").write(code)
    print("wrote", out_path, len(code.splitlines()), "lines")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
