"""CLI for the v13 revision runs (batch APIs only).

  python -m coralagent.run check                  # offline checks: data, images, keys present, imports
  python -m coralagent.run setup-images           # download CoralVQA_Image.zip from HF, extract the 500 bank images
  python -m coralagent.run stage E0 --dry-run     # count requests and worst-case cost; no API calls
  python -m coralagent.run stage E0               # run a stage (must be allowed in control/CONTROL.json)
  python -m coralagent.run status                 # spend, open batches, result counts
  python -m coralagent.run export                 # write results/v13/*.jsonl.gz and reports/status.json

Safety rules (PROTOCOL.md section 6): worst-case budget check before every batch; 10-request test batch
per provider and stage before anything larger; batch ids saved before polling (no double submission);
at most one resubmission of failed requests; stop when a round has more than 5% failures.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from . import config, gateway, runtime, tasks
from .store import Store

core = runtime.core
EXPECTED_OUT = {"baseline_strict": 30, "baseline_reefbank": 700, "tool": 500, "agent": 500, "verifier": 1100,
                "single_call": 500, "agent_generic": 400, "judge_evidence": 300}


class Stop(Exception):
    pass


# ================================ row sets ========================================
def h100_image_indices() -> List[int]:
    rng = random.Random(2026)
    return sorted(rng.sample(list(range(100, 500)), 100))


def image_subset(indices: List[int], n: int, seed: int = 2026) -> List[int]:
    """Pre-registered image subset (budget fallback): n images drawn with a fixed seed."""
    return sorted(random.Random(seed).sample(sorted(indices), n))


def subset_rows(rows, n_images):
    if not n_images:
        return rows
    keep = image_subset(sorted(set(rows["active_image_index"])), int(n_images))
    return rows[rows["active_image_index"].isin(keep)]


def rowsets(df) -> Dict[str, Any]:
    D = df[df["split"] == "D"]
    H = df[df["split"] == "H"]
    H100 = H[H["active_image_index"].isin(h100_image_indices())]
    e0 = D.sort_values(["active_image_index", "pilot_question_index"]).head(50)
    return {"D": D, "H": H, "ALL": df, "H100": H100, "E0": e0,
            "H_genus": H[H["question_type"] == "genus_taxonomy"],
            "H100_genus": H100[H100["question_type"] == "genus_taxonomy"]}


# ================================ stages ==========================================
def stage_arms(stage: str, rs: Dict[str, Any], store: Store, opts: Dict[str, Any]) -> List[Tuple[str, Callable, Any]]:
    T = tasks
    if stage == "E0":
        e0 = rs["E0"]
        return [("A3v13_trace", lambda r: T.trace(r, "A3v13_trace"), e0),
                ("E5_single_call_same_info", lambda r: T.single_call(r), e0.head(15)),
                ("E7a_generic_gpt5_agent", lambda r: T.generic_agent(r), e0.head(15)),
                ("E0_gemma_probe", lambda r: T.reefbank(r, "E0_gemma_probe", "together", config.GEMMA), e0.head(10))]
    if stage == "E1":
        arms = [("B0b_strict_v13", lambda r: T.strict_gpt5(r, "B0b_strict_v13"), rs["H"]),
                ("B2_grok_v13", lambda r: T.reefbank(r, "B2_grok_v13", "xai", config.GROK), rs["H"]),
                ("B4_gemma_v13", lambda r: T.reefbank(r, "B4_gemma_v13", "together", config.GEMMA), rs["H"])]
        if opts.get("include_gpt41"):
            arms.append(("B1_gpt41_v13", lambda r: T.reefbank(r, "B1_gpt41_v13", "openai", config.GPT41), rs["H"]))
        return arms
    if stage == "E2":
        runtime.load_prior(store.results("B0b_strict_v13"))
        return [("A3v13_trace", lambda r: T.trace(r, "A3v13_trace"), rs["ALL"])]
    if stage in ("E3", "E3b"):   # E3b: E3 extended to all 100 D images (deviation 2026-11-17)
        runtime.load_prior(store.results("B0b_strict_v13"))
        names = ["A0v13_scope_only", "A1v13_prior_scope", "A2v13_fixed_all_tools", "A4v13_no_retrieval",
                 "A6v13_no_prior", "A3v13_no_scope"] + (["A7v13_plus_bleaching_tools"] if opts.get("include_a7", True) else [])
        n_img = None if stage == "E3b" else opts.get("e3_images")
        if n_img == "auto":   # budget-only rule, fixed in PROTOCOL.md before E3 (2026-11-16)
            left = config.caps()["openai"] - store.spent("openai") - store.reserved("openai")
            n_img = None if left >= 11.0 else (50 if left >= 5.5 else 25)
            log(f"E3 size rule: ${left:.2f} OpenAI budget left -> {n_img or 100} D images")
        d = subset_rows(rs["D"], n_img)
        return [(n, (lambda n: (lambda r: T.trace(r, n)))(n), d) for n in names]
    if stage == "E4":
        runtime.load_prior(store.results("B0b_strict_v13"))
        return [("B0b_strict_genuslist_v13", lambda r: T.strict_gpt5(r, "B0b_strict_genuslist_v13", genus_list=True), rs["H_genus"]),
                ("B2_grok_genuslist_v13", lambda r: T.reefbank(r, "B2_grok_genuslist_v13", "xai", config.GROK, genus_list=True), rs["H_genus"]),
                ("B4_gemma_genuslist_v13", lambda r: T.reefbank(r, "B4_gemma_genuslist_v13", "together", config.GEMMA, genus_list=True), rs["H_genus"]),
                ("A3v13_no_genus_list", lambda r: T.trace(r, "A3v13_no_genus_list"), rs["H100_genus"])] + (
                [("B1_gpt41_genuslist_v13", lambda r: T.reefbank(r, "B1_gpt41_genuslist_v13", "openai", config.GPT41, genus_list=True),
                  rs["H_genus"])] if opts.get("include_gpt41") else [])
    if stage == "E5":
        return [("E5_single_call_same_info", lambda r: T.single_call(r), rs["H100"])]
    if stage == "E6":
        k = int(opts.get("e6_k", 3))
        return [("E6_self_consistency", lambda r: T.self_consistency(r, k), rs["H100"])]
    if stage == "E7a":
        runtime.load_prior(store.results("B0b_strict_v13"))
        return [("E7a_generic_gpt5_agent", lambda r: T.generic_agent(r), subset_rows(rs["H100"], opts.get("e7a_images")))]
    if stage in ("E8", "E8b"):   # E8b: second, independent judge (GPT-4.1) for judge agreement
        a3 = {r["sample_id"]: r for r in store.results("A3v13_trace")}
        ag = {r["sample_id"]: r for r in store.results("E7a_generic_gpt5_agent")}
        h = rs["H100"]
        h = h[h["sample_id"].isin(set(a3) & set(ag))]
        h = subset_rows(h, opts.get("e8_images"))
        judge = opts.get("judge_b", "gpt-4.1") if stage == "E8b" else opts.get("judge")
        jp, jm = {"gpt-4.1": ("openai", config.GPT41), "grok": ("xai", config.GROK)}.get(
            judge, ("together", config.GEMMA))
        na, ne = (("E8b_judge2_A3v13", "E8b_judge2_E7a") if stage == "E8b" else ("E8_judge_A3v13", "E8_judge_E7a"))
        return [(na, lambda r: T.judge(r, a3[r["sample_id"]], na, jp, jm), h),
                (ne, lambda r: T.judge(r, ag[r["sample_id"]], ne, jp, jm), h)]
    raise SystemExit(f"unknown stage {stage}")


# ================================ round loop ======================================
def replay(stage: str, arms, store: Store) -> Dict[str, int]:
    gateway.PENDING.clear()
    gateway.STAGE["name"] = stage
    stats = Counter()
    for method, fn, rows in arms:
        for _, row in rows.iterrows():
            if store.has_result(method, row["sample_id"]):
                stats["done"] += 1
                continue
            try:
                rec = fn(row)
                store.put_result(method, row["sample_id"], rec)
                stats["done"] += 1
                stats["new_done"] += 1
            except gateway.NeedCall:
                stats["pending_rows"] += 1
            except gateway.WaitDependency:
                stats["waiting_rows"] += 1
        store.commit()
    stats["pending_requests"] = len(gateway.PENDING)
    return dict(stats)


def by_provider(pending: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for k, s in pending.items():
        out[s["provider"]][k] = s
    return out


def cost_summary(specs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    worst = sum(gateway.worst_cost(s) for s in specs.values())
    exp = 0.0
    for s in specs.values():
        t = gateway.estimate_tokens(s)
        pin, pout = config.PRICES.get(s["model"], (5, 15))
        out = min(t["out_worst"], EXPECTED_OUT.get(s["purpose"], 500))
        exp += (t["in"] * pin + out * pout) / 1e6 * config.BATCH_FACTOR.get(s["provider"], 1.0)
    return {"n": len(specs), "worst_usd": round(worst, 4), "expected_usd": round(exp, 4)}


def remaining(store: Store, provider: str) -> float:
    return config.caps()[provider] - store.spent(provider) - store.reserved(provider)


def submit(store: Store, provider: str, stage: str, kind: str, specs: Dict[str, Dict[str, Any]], adapter) -> None:
    worst = sum(gateway.worst_cost(s) for s in specs.values())
    if worst > remaining(store, provider) + 1e-9:
        raise Stop(f"budget guard: {provider} {kind} batch worst-case ${worst:.2f} exceeds remaining "
                   f"${remaining(store, provider):.2f}")
    name = f"coral_{stage}_{kind}_{int(time.time())}"
    batch_id, meta = adapter.submit(specs, name)
    store.add_batch(batch_id, provider, stage, kind, list(specs), worst, meta)    # saved before any polling
    log(f"submitted {provider} {kind} batch {batch_id}: {len(specs)} requests, worst-case ${worst:.2f}")


def chunk_by_budget(store: Store, provider: str, specs: Dict[str, Dict[str, Any]],
                    max_n: int = 1000) -> List[Dict[str, Dict[str, Any]]]:
    """One batch per provider per round: at most max_n requests and never more than the remaining budget
    (worst case). The rest is picked up in the next round, after actual costs are known."""
    room = remaining(store, provider)
    chunk, total = {}, 0.0
    for k, s in specs.items():
        w = gateway.worst_cost(s)
        if total + w > room or len(chunk) >= max_n:
            break
        chunk[k] = s; total += w
    if not chunk:
        raise Stop(f"budget guard: no room for even one {provider} request (remaining ${room:.2f})")
    return [chunk]


def wait_and_ingest(store: Store) -> Dict[str, Counter]:
    """Poll every open batch until done; ingest results; return per-provider ok/bad counts."""
    quality: Dict[str, Counter] = defaultdict(Counter)
    while True:
        open_b = store.open_batches()
        if not open_b:
            return quality
        for b in open_b:
            ad = providers_adapter(b["provider"])
            try:
                st = ad.status(b["batch_id"], b["meta"])
            except Exception as e:                       # transient API/network error: retry next poll
                log(f"status check failed for {b['batch_id']}: {e!r}")
                continue
            age_h = (time.time() - float(b["meta"].get("_created", time.time()))) / 3600
            if st == "running" and age_h > config.MAX_BATCH_WAIT_HOURS:
                log(f"batch {b['batch_id']} still running after {age_h:.1f} h: cancelling, treating as expired")
                try:
                    if b["provider"] == "openai":
                        for i in b["meta"].get("ids", []):
                            ad.client.batches.cancel(i)
                except Exception as e:
                    log(f"cancel failed: {e!r}")
                st = "expired"
            if st == "running":
                continue
            results = ad.fetch(b["batch_id"], b["meta"])
            got = set()
            total = 0.0
            for res in results:
                key = res["key"]
                if key not in b["keys"]:
                    continue
                got.add(key)
                spec = store.get_spec(key) or {}
                cost = gateway.actual_cost(b["provider"], spec.get("model", ""), res.get("usage"), res.get("cost_usd"))
                total += cost
                store.put_response(key, b["provider"], res["status"], res.get("text") or "", res.get("usage"), cost,
                                   res.get("model"), b["batch_id"], res.get("raw"))
                quality[b["provider"]]["bad" if gateway.is_bad(spec, res["status"], res.get("text") or "") else "ok"] += 1
            if set(b["keys"]) - got or st != "done":
                err = ""
                try:
                    err = ad.batch_error(b["batch_id"], b["meta"]) if hasattr(ad, "batch_error") else ""
                except Exception as e:
                    err = f"could not read batch error: {e!r}"
                with open(config.REPORTS / "provider_errors.md", "a", encoding="utf-8") as f:
                    f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} {b['provider']} {b['kind']} {b['batch_id']} "
                            f"(status {st}, {len(got)}/{len(b['keys'])} lines)\n\n```\n{err[:4000]}\n```\n")
                log(f"batch {b['batch_id']} problem written to reports/provider_errors.md")
            for key in set(b["keys"]) - got:                      # missing lines count as errors
                store.put_response(key, b["provider"], "error", "", None, 0.0, None, b["batch_id"], {"missing": True})
                quality[b["provider"]]["bad"] += 1
            store.close_batch(b["batch_id"], "done" if st == "done" else st, total)
            log(f"batch {b['batch_id']} ({b['provider']} {b['kind']}) finished: {len(got)}/{len(b['keys'])} lines, "
                f"actual ${total:.4f}")
        if store.open_batches():
            time.sleep(config.POLL_SECONDS)


_ADAPTERS: Dict[str, Any] = {}


def providers_adapter(provider: str):
    if MOCK["on"]:
        from .mock import MockAdapter
        return _ADAPTERS.setdefault(provider, MockAdapter(provider))
    from . import providers
    return providers.adapter(provider)


MOCK = {"on": False}


def write_bad_responses(stage: str, store: Store, examples: int = 4) -> str:
    """No-spend diagnostics: failed responses of a stage grouped by model and purpose, with examples."""
    rows = store.db.execute(
        "SELECT q.key, q.model, q.spec, r.status, r.attempts, r.text, r.usage, r.raw FROM requests q "
        "JOIN responses r ON r.key = q.key WHERE q.stage = ?", (stage,)).fetchall()
    groups: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    totals: Counter = Counter()
    for key, model, spec_s, status, attempts, text, usage_s, raw_s in rows:
        spec = json.loads(spec_s)
        g = (model, spec.get("purpose") or "?")
        totals[g] += 1
        if gateway.is_bad(spec, status, text):
            groups[g].append((key, spec, status, attempts, text, usage_s, raw_s))
    out = [f"# {stage}: failed responses", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
           "| Model | Purpose | Failed | Total |", "|---|---|---|---|"]
    for g in sorted(totals):
        out.append(f"| {g[0]} | {g[1]} | {len(groups.get(g, []))} | {totals[g]} |")
    for g, items in sorted(groups.items()):
        out += ["", f"## {g[0]} / {g[1]}"]
        reasons = Counter()
        for _, _, status, _, text, _, raw_s in items:
            try:
                body = ((json.loads(raw_s or "{}").get("response") or {}).get("body") or {})
                reasons[f"status={status} body_status={body.get('status')} {body.get('incomplete_details')}"] += 1
            except Exception:
                reasons[f"status={status}"] += 1
        out += [f"- {n}x {r}" for r, n in reasons.most_common(6)]
        for key, spec, status, attempts, text, usage_s, _ in items[:examples]:
            out += ["", f"### {key} (attempts {attempts}, status {status})", "",
                    f"max_tokens={spec.get('max_tokens')} effort={spec.get('effort')} expect_json={spec.get('expect_json')}",
                    f"usage: {str(usage_s)[:400]}", "", "Prompt (end):", "```", str(spec.get('prompt'))[-600:], "```",
                    "Output:", "```", str(text)[:800], "```"]
    path = config.REPORTS / f"{stage}_bad_responses.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return str(path)


def run_stage(stage: str, dry_run: bool = False, max_rounds: int = 30) -> Dict[str, Any]:
    ctrl = config.load_control()
    allowed = stage in (ctrl.get("allowed_stages") or []) or ((ctrl.get("stages") or {}).get(stage) or {}).get("allow")
    if not dry_run and not allowed and not MOCK["on"]:
        raise SystemExit(f"Stage {stage} is not allowed by control/CONTROL.json (allowed: {ctrl.get('allowed_stages')}).")
    store = Store()
    gateway.STORE = store
    df = runtime.init()
    rs = rowsets(df)
    opts = ctrl.get("options") or {}
    summary = {"stage": stage, "rounds": [], "stopped": None}
    if stage in (opts.get("diagnose_only") or []) and not dry_run:
        path = write_bad_responses(stage, store)
        summary["stopped"] = f"diagnose only (no spend): see {path}"
        log(f"[{stage}] diagnose only: wrote {path}")
        write_stage_report(stage, store, summary)
        return summary
    arms = stage_arms(stage, rs, store, opts)
    for rnd in range(1, max_rounds + 1):
        if store.open_batches() and not dry_run:      # resume after a crash: never resubmit, wait instead
            log(f"[{stage}] waiting for {len(store.open_batches())} batch(es) submitted earlier")
            wait_and_ingest(store)
        st = replay(stage, arms, store)
        pend = by_provider(dict(gateway.PENDING))
        disabled = {p for p in list(pend) if store.get_kv(f"disabled:{stage}:{p}")}
        if disabled:
            summary["skipped_providers"] = {p: store.get_kv(f"disabled:{stage}:{p}") for p in disabled}
            for p in disabled:
                pend.pop(p, None)
        cs = {p: cost_summary(s) for p, s in pend.items()}
        log(f"[{stage}] round {rnd}: {st} | pending by provider: {cs}")
        summary["rounds"].append({"round": rnd, "stats": st, "pending": cs})
        if dry_run:
            summary["dry_run_example_requests"] = _examples(pend)
            break
        if not pend:
            if st.get("waiting_rows"):
                summary["stopped"] = f"{st['waiting_rows']} rows wait for a dependency (e.g. the E1 strict prior)"
            break
        try:
            # 1) test batch first (once per stage and provider)
            for p, specs in pend.items():
                flag = f"canary_ok:{stage}:{p}"
                if not store.get_kv(flag):
                    canary = dict(list(specs.items())[: config.CANARY_SIZE])
                    submit(store, p, stage, "canary", canary, providers_adapter(p))
            if any(not store.get_kv(f"canary_ok:{stage}:{p}") for p in pend):
                autopush(store, stage, summary, f"run: {stage} test batches submitted")
                q = wait_and_ingest(store)
                autopush(store, stage, summary, f"run: {stage} test batches finished")
                for p in pend:
                    flag = f"canary_ok:{stage}:{p}"
                    if store.get_kv(flag):
                        continue
                    if q[p]["bad"] > 0 and f"{stage}:{p}" not in (ctrl.get("accept_canary") or []):
                        if p in (ctrl.get("optional_providers") or []) and store.get_kv(f"canary_fail:{stage}:{p}"):
                            store.set_kv(f"disabled:{stage}:{p}", f"test batch failed twice: {dict(q[p])}")
                            log(f"{p} disabled for {stage} (optional provider, test batch failed twice); continuing without it")
                            continue
                        store.set_kv(f"canary_fail:{stage}:{p}", True)
                        if p in (ctrl.get("optional_providers") or []):
                            log(f"test batch failed for optional provider {p}; it gets one more test batch")
                            continue
                        raise Stop(f"test batch failed for {p}: {dict(q[p])} — see reports; no further spend")
                    store.set_kv(flag, True)
                    log(f"test batch passed for {p}: {dict(q[p])}")
                continue          # replay: canary responses are now in the store
            # 2) full submission within the budget
            for p, specs in pend.items():
                for chunk in chunk_by_budget(store, p, specs, int(opts.get("max_requests_per_batch", 1000))):
                    submit(store, p, stage, "round%d" % rnd, chunk, providers_adapter(p))
            autopush(store, stage, summary, f"run: {stage} round {rnd} submitted")
            q = wait_and_ingest(store)
            autopush(store, stage, summary, f"run: {stage} round {rnd}")
            for p, c in q.items():
                n = c["ok"] + c["bad"]
                # tiny tail rounds (a few retried requests) are too small for a rate:
                # there the rule needs at least STOP_MIN_BAD failures
                if n and c["bad"] / n > config.STOP_FAILURE_RATE and \
                        (n >= config.STOP_MIN_N or c["bad"] >= config.STOP_MIN_BAD):
                    raise Stop(f"stop rule: {p} round {rnd} failure rate {c['bad']}/{n} > 5%")
        except Stop as e:
            summary["stopped"] = str(e)
            try:
                write_bad_responses(stage, store)
            except Exception as ex:          # diagnostics must never mask the stop
                log(f"bad-response report failed: {ex}")
            log("STOPPED: " + str(e))
            break
    export(store)
    write_stage_report(stage, store, summary)
    (config.REPORTS / f"{stage}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def autopush(store: Store, stage: str, summary: Dict[str, Any], message: str) -> None:
    """Progress reports during long stages (lab daemon only)."""
    if os.environ.get("CORAL_AUTOPUSH") != "1":
        return
    try:
        export(store)
        write_stage_report(stage, store, summary)
        from .daemon import git_sync
        git_sync(message, paths=("reports",))
    except Exception as e:  # reporting must never break a run
        log(f"autopush failed: {e!r}")


def _examples(pend) -> Dict[str, Any]:
    out = {}
    for p, specs in pend.items():
        k, s = next(iter(specs.items()))
        s2 = {kk: (v if kk != "raw_body" else "<agent body>") for kk, v in s.items()}
        s2["prompt"] = (s2.get("prompt") or "")[:400]
        out[p] = {"key": k, "spec": s2}
    return out


# ================================ reporting =======================================
def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(line, flush=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def spend_table(store: Store) -> Dict[str, Dict[str, float]]:
    caps = config.caps()
    return {p: {"spent_usd": round(store.spent(p), 4), "reserved_usd": round(store.reserved(p), 4), "cap_usd": caps[p]}
            for p in caps}


def export(store: Store) -> None:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    counts = store.result_counts()
    for method in counts:
        rows = store.results(method)
        with gzip.open(config.RESULTS / f"{method}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    # A3 without GUARD, derived from the same trajectories (no calls)
    a3 = store.results("A3v13_trace")
    if a3:
        with gzip.open(config.RESULTS / "A5v13_no_guard.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in a3:
                f.write(json.dumps(tasks.trace_no_guard_from(r), ensure_ascii=False, default=str) + "\n")
    usage = defaultdict(lambda: Counter())
    for key, stage, prov, model, status, attempts, u, cost, mret in store.usage_rows():
        c = usage[f"{stage}|{prov}|{model}"]
        c["responses"] += 1; c["ok"] += status == "ok"; c["cost_usd"] += cost or 0.0
        c["retried"] += (attempts or 1) > 1
        if mret:
            c[f"model_returned={mret}"] += 1
    status = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "spend": spend_table(store), "results": counts,
              "usage": {k: dict(v) for k, v in usage.items()}, "open_batches": store.open_batches()}
    (config.REPORTS / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def write_stage_report(stage: str, store: Store, summary: Dict[str, Any]) -> None:
    lines = [f"# Stage {stage} report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## Outcome"); lines.append("")
    lines.append("STOPPED: " + summary["stopped"] if summary.get("stopped") else "Completed (no stop condition).")
    lines += ["", "## Spend (USD)", "", "| Provider | Spent | Reserved | Cap |", "|---|---|---|---|"]
    for p, v in spend_table(store).items():
        lines.append(f"| {p} | {v['spent_usd']:.4f} | {v['reserved_usd']:.4f} | {v['cap_usd']:.2f} |")
    lines += ["", "## Rounds", ""]
    for r in summary["rounds"]:
        lines.append(f"- Round {r['round']}: {json.dumps(r['stats'])}; pending {json.dumps(r['pending'])}")
    lines += ["", "## Batches", "", "| Batch | Provider | Kind | n | Status | Worst | Actual |", "|---|---|---|---|---|---|---|"]
    for b in store.batches_for_stage(stage):
        lines.append(f"| {b['batch_id']} | {b['provider']} | {b['kind']} | {b['n']} | {b['status']} | "
                     f"{(b['worst'] or 0):.3f} | {(b['actual'] or 0):.4f} |")
    lines += ["", "## Results so far", "", "| Method | Rows |", "|---|---|"]
    for m, n in sorted(store.result_counts().items()):
        lines.append(f"| {m} | {n} |")
    if summary.get("dry_run_example_requests"):
        lines += ["", "## Dry run: example request per provider", "", "```json",
                  json.dumps(summary["dry_run_example_requests"], indent=2, default=str)[:6000], "```"]
    (config.REPORTS / f"{stage}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ================================ setup / checks ==================================
def setup_images() -> None:
    from huggingface_hub import hf_hub_download
    df = runtime.load_bank()
    names = sorted({Path(p).name for p in df["local_image_path"]})
    config.IMAGES.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (config.IMAGES / n).exists()]
    if missing:
        zpath = hf_hub_download("CoralReefData/CoralVQA", "CoralVQA_Image.zip", repo_type="dataset",
                                cache_dir=str(config.DATA / "hf_cache"))
        with zipfile.ZipFile(zpath) as z:
            members = set(z.namelist())
            for n in missing:
                if n not in members:
                    raise SystemExit(f"{n} not in zip")
                (config.IMAGES / n).write_bytes(z.read(n))
    manifest = json.loads((config.REPO / "reference" / "image_sha1_manifest.json").read_text())
    bad = []
    for n in names:
        h = hashlib.sha1((config.IMAGES / n).read_bytes()).hexdigest()
        if manifest.get(n) != h:
            bad.append(n)
    print(f"images: {len(names)} required, {len(names) - len(bad)} byte-identical to the v12 inputs, {len(bad)} mismatched")
    if bad:
        raise SystemExit(f"mismatched images: {bad[:5]}")


def check() -> None:
    ok = True
    for p in [config.CORALVQA_JSONL, config.BANK_CSV, config.V12_BASELINES, config.IMAGES]:
        print(("OK  " if p.exists() else "MISSING ") + str(p)); ok &= p.exists()
    for k in ["OPENAI_API_KEY", "XAI_API_KEY", "TOGETHER_API_KEY"]:
        print(f"{k}: {'present' if os.environ.get(k) else 'MISSING'}"); ok &= bool(os.environ.get(k))
    df = runtime.init()
    n_img = sum(Path(p).exists() for p in set(df["local_image_path"]))
    print(f"bank rows {len(df)}, images present {n_img}/500, genus labels {len(core.KNOWN_GENUS_LIST)}, prior rows {len(runtime.PRIOR)}")
    print("control:", json.dumps(config.load_control()))
    print("ALL CHECKS PASSED" if ok and n_img == 500 else "CHECKS FAILED")


def load_env() -> None:
    env = config.REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main(argv=None) -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "setup-images", "stage", "status", "export"])
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline test: fake provider responses, no network")
    a = ap.parse_args(argv)
    config.ensure_dirs()
    MOCK["on"] = a.mock
    if a.cmd == "check":
        check()
    elif a.cmd == "setup-images":
        setup_images()
    elif a.cmd == "stage":
        print(json.dumps(run_stage(a.stage, dry_run=a.dry_run), indent=2, default=str)[:4000])
    elif a.cmd in {"status", "export"}:
        store = Store(); gateway.STORE = store
        export(store)
        print((config.REPORTS / "status.json").read_text())


if __name__ == "__main__":
    main()
