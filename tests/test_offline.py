"""Offline checks (no API calls). Run on the lab PC before the pilot:

    .venv/bin/python -m tests.test_offline

Uses the mock provider for the pipeline and safety-rule tests, in a temporary run directory.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="coral_test_"))
    from coralagent import config
    real_root = config.ROOT
    # isolate all run state in a temp copy of the outputs location
    config.RUN = tmp / "v13"; config.STORE_DB = config.RUN / "llm_store.sqlite"
    config.BATCH_DIR = config.RUN / "batch_files"; config.WORK = config.RUN / "work"
    config.RESULTS = tmp / "results"; config.REPORTS = tmp / "reports"; config.POLL_SECONDS = 0
    from coralagent import evaluation as ev, gateway, mock, run, runtime, tasks
    from coralagent.store import Store

    df = runtime.init()
    rs = run.rowsets(df)
    assert len(df) == 3000 and (df["split"] == "D").sum() == 600, "bank size"
    assert len(rs["H100"]) == 600 and len(rs["E0"]) == 50
    assert len(runtime.PRIOR) >= 590, "strict GPT-5 prior for images 1-100"
    print("bank, splits, prior: OK")

    # fix 3: colour questions mentioning a genus no longer route to the genus tool
    col = df[(df["question_type"] == "color_visual") & df["question"].str.lower().str.contains("genus")]
    n_v12 = sum(runtime.v12_question_intent(r)["asks_genus"] for _, r in col.iterrows())
    n_v13 = sum(runtime.core.question_intent(r)["asks_genus"] for _, r in col.iterrows())
    print(f"colour questions mentioning 'genus': {len(col)}; routed to genus tool v12={n_v12} v13={n_v13}")
    assert n_v13 == 0

    # labels (comment 7)
    lab = [ev.label(r.question, r.question_type, r.answer_gt) for r in df.itertuples()]
    n_task = sum(ev.in_visible_bleaching_task(r.question, r.question_type) for r in df.itertuples())
    assert n_task == 412 and lab.count("visible_bleaching") == 66 and lab.count("no_visible_bleaching") == 346
    print("visible-bleaching labels: 412 questions, 66 positive, 346 negative: OK")

    # request settings: fix 1 for tools, strict baseline unchanged
    store = Store(); gateway.STORE = store; gateway.PENDING.clear()
    row = rs["E0"].iloc[0]
    for fn in (lambda: tasks.trace(row, "A3v13_trace"), lambda: tasks.strict_gpt5(row, "t")):
        try:
            fn()
        except gateway.NeedCall:
            pass
    specs = list(gateway.PENDING.values())
    tool = [s for s in specs if s["purpose"] == "tool"]
    strict = [s for s in specs if s["purpose"] == "baseline_strict"]
    assert tool and all(s["effort"] == "low" and s["max_tokens"] >= 2000 for s in tool)
    assert strict and all(s["effort"] == "low" and s["max_tokens"] >= 2000 for s in strict)
    assert gateway.spec_key(tool[0]) == gateway.spec_key(dict(tool[0])), "deterministic keys"
    print("request settings (all GPT-5 calls low effort / 2000 tokens): OK")

    # pipeline + safety rules with the mock provider
    run.MOCK["on"] = True
    s = run.run_stage("E0")
    assert not s["stopped"], s["stopped"]
    print("E0 end-to-end on mock provider: OK", store.result_counts() if False else "")
    shutil.rmtree(config.RUN); run._ADAPTERS.clear(); config.ensure_dirs()
    run._ADAPTERS["openai"] = mock.MockAdapter("openai", fail_rate=0.5, seed=1)
    s = run.run_stage("E0")
    assert s["stopped"] and "test batch failed" in s["stopped"]
    print("stop on failing test batch: OK")
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OFFLINE TESTS PASSED")


def test_openai_batch_files_are_single_model(tmp_path, monkeypatch):
    """OpenAI rejects mixed-model batch files (E1 round 5: 358 'mismatched_model' lines)."""
    import json as _json
    from types import SimpleNamespace
    from coralagent import config as _cfg
    from coralagent.providers import OpenAIBatch
    monkeypatch.setattr(_cfg, "BATCH_DIR", tmp_path)
    created = []

    class _Files:
        def create(self, file, purpose):
            created.append(file.read().decode()); return SimpleNamespace(id=f"f{len(created)}")

    class _Batches:
        def create(self, **kw):
            return SimpleNamespace(id=f"b{len(created)}")

    a = OpenAIBatch.__new__(OpenAIBatch)
    a.client = SimpleNamespace(files=_Files(), batches=_Batches())
    specs = {}
    for i in range(7):
        m = "gpt-5" if i % 2 else "gpt-4.1"
        specs[f"k{i}"] = {"model": m, "prompt": "q", "max_tokens": 10, "images": []}
    bid, meta = a.submit(specs, "t")
    assert len(meta["ids"]) == 2
    for text in created:
        models = {_json.loads(l)["body"]["model"] for l in text.splitlines()}
        assert len(models) == 1
    assert sum(len(t.splitlines()) for t in created) == 7


if __name__ == "__main__":
    main()
