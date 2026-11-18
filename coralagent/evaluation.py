"""Pre-registered evaluation (PROTOCOL.md sections 4-5). Frozen before any v13 result exists.

- Accuracy: the unchanged v12 task-aware scorer (core.quick_semantic_score), score >= 1 counts as correct.
- Visible-bleaching task (comment 7): only CoralVQA bleaching/health questions that explicitly ask whether
  bleaching is visible (or how many bleached colonies are visible). Susceptibility questions, colour,
  mortality, disease and generic health questions are excluded. The same rule maps reference answers and
  predicted answers, for every method alike.
- Paired comparison: accuracy difference with a 95% bootstrap CI resampling whole images, plus an exact
  McNemar test on discordant pairs. Holm correction for the secondary family.
"""
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUSCEPTIBILITY = re.compile(r"suscept|prone|known for|resist|vulnerab|sensitiv|likely to|risk|tend to|tolera")
BLEACH = re.compile(r"bleach")
NUM = re.compile(r"-?\d+(?:\.\d+)?")
NUMBER_WORDS = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "several": 3,
                "many": 5, "multiple": 3}


def in_visible_bleaching_task(question: str, question_type: str) -> bool:
    q = str(question or "").lower()
    return question_type == "bleaching_health" and bool(BLEACH.search(q)) and not SUSCEPTIBILITY.search(q)


def bleaching_class(answer: Any) -> Optional[str]:
    """Map an answer to the two classes; None = no verdict (unclear, empty, off-format)."""
    a = re.sub(r"[^a-z0-9. ]+", " ", str(answer or "").lower()).strip()
    if not a:
        return None
    first = a.split()[0].rstrip(".")
    if first in {"yes", "true", "present"}:
        return "visible_bleaching"
    if first in {"no", "false", "none", "absent", "zero"}:
        return "no_visible_bleaching"
    m = NUM.search(a)
    if m:
        return "visible_bleaching" if float(m.group(0)) > 0 else "no_visible_bleaching"
    if first in NUMBER_WORDS:
        return "visible_bleaching" if NUMBER_WORDS[first] > 0 else "no_visible_bleaching"
    if re.search(r"\bno (visible )?bleach|\bnot bleached|\bhealthy\b|\bunbleached", a):
        return "no_visible_bleaching"
    if re.search(r"\bbleached\b|\bbleaching (is )?(visible|present|evident)", a):
        return "visible_bleaching"
    return None


def label(question: str, question_type: str, answer_gt: Any) -> Optional[str]:
    return bleaching_class(answer_gt) if in_visible_bleaching_task(question, question_type) else None


def prediction(question: str, question_type: str, answer_pred: Any) -> Optional[str]:
    return bleaching_class(answer_pred) if in_visible_bleaching_task(question, question_type) else None


# ---------------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(rows: Sequence[Tuple[str, float, float]], n_boot: int = 10000, seed: int = 2026
                     ) -> Dict[str, float]:
    """rows: (image_id, correct_A, correct_B). Resamples whole images. Returns diff A-B and 95% CI."""
    by_img: Dict[str, List[Tuple[float, float]]] = {}
    for img, a, b in rows:
        by_img.setdefault(img, []).append((a, b))
    imgs = list(by_img)
    tot = [(sum(x for x, _ in by_img[i]), sum(y for _, y in by_img[i]), len(by_img[i])) for i in imgs]
    n = sum(t[2] for t in tot)
    diff = (sum(t[0] for t in tot) - sum(t[1] for t in tot)) / n
    rng = random.Random(seed)
    ds = []
    for _ in range(n_boot):
        s = [tot[rng.randrange(len(tot))] for _ in tot]
        nn = sum(t[2] for t in s)
        ds.append((sum(t[0] for t in s) - sum(t[1] for t in s)) / nn)
    ds.sort()
    return {"diff": diff, "ci_low": ds[int(0.025 * n_boot)], "ci_high": ds[int(0.975 * n_boot) - 1],
            "n_questions": n, "n_images": len(imgs)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
