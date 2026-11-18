# CoralAgent-TRACE v13 — pre-registered revision protocol

Frozen before any v13 API call. Any later change is recorded in `## Deviations` with its date and reason.
Manuscript 1952393 (Frontiers in Marine Science), Reviewer 1, round 1.

## 1. What changes from v12 (implementation fixes only)
v13 reuses the v12 notebook logic unchanged. `coralagent/v12_core.py` is generated mechanically from
`reference/CoralAgent_TRACE_v12.ipynb` (public repo commit 30fa7e7). The only changes, in `coralagent/runtime.py`:

1. **Output budget.** GPT-5 tool, finalizer and repair calls use `reasoning.effort = "low"` and at least 2,000
   output tokens. Under v12, reasoning used up the 350–750-token limit:
   - genus identification was empty on 599 of 600 development calls;
   - focused VQA was empty on about 60% of calls;
   - the morphology summary was empty on about 72% of calls.
2. **Baseline prior.** The prior is the strict GPT-5 zero-shot answer (B0b). The deprecated JSON GPT-5 run (B0) had
   empty answers on 32% of development rows and 71% of extension rows.
3. **Answer-type priority (comment 10).** For CoralVQA colour and morphology questions, a "genus" keyword no longer
   routes the question to the genus tool or sets the expected answer style to genus (Eq. 5). v12 routed 211 of the
   378 colour questions that mention "genus" to the genus tool; v13 routes none.
4. **Ablations.** Every ablation is built on A3, the proposed system with the bleaching tools off. Each one removes
   (or, for A7, adds) exactly one component.
5. **No cross-provider fallback.** A failed request gets one resubmission to the same model, then counts as a
   technical failure. v12 silently fell back from GPT-5 to Grok.

Unchanged: all prompts, thresholds, arbitration rules, the GUARD verifier (Grok-4.3, 750 tokens), protected
repair, retrieval, SCOPE crops, the genus list in prompts, and the v12 scorer.

## 2. Data
- **Source:** `CoralVQA_train.jsonl` from Hugging Face `CoralReefData/CoralVQA` (the labelled train-format file;
  not the official test or cross-region split).
- **Bank:** the persistent 500-image, 3,000-question bank from v12
  (`active_pilot_500images_v12_TRACE_bank500_q6.csv`, all IDs released).
- **Images:** byte-identical to the v12 inputs (`reference/image_sha1_manifest.json`).
- **D (development):** images with bank index 0–99 (600 questions). Used for ablations only.
- **H (held-out):** bank index 100–499 (2,400 questions). Used for the main comparison.
- **H100:** 100 H images drawn with `random.Random(2026).sample(range(100, 500), 100)` (600 questions), listed in
  `reference/H100_image_indices.json`.

## 3. Experiments
| ID | Arm(s) | Rows | Reviewer comment |
|---|---|---|---|
| E0 | Pilot of all request types (no reported results) | 50 D | — |
| E1 | Strict GPT-5 (v12 settings), Grok-4.3 and Gemma-4-31B (v12 ReefBank prompt); GPT-4.1 optional | H | 3 |
| E2 | A3v13 (the proposed system) | D + H | 3, 5, 8, 10, 13 |
| E3 | A0 SCOPE only; A1 prior+SCOPE; A2 fixed all tools; A4 −retrieval; A6 −prior; A3 −SCOPE; A7 +bleaching tools; A5 −GUARD (derived from the A3 trajectories, no calls) | D | 5 |
| E4 | Baselines given the same genus list; A3v13 without the genus list (prompts and validation) | H genus / H100 genus | 4 |
| E5 | One GPT-5 call (low effort) with the full image, the SCOPE crop, the top-5 retrieved records, the same answer rules and the genus list | H100 | 6 |
| E6 | Strict GPT-5, k=3 samples (one from E1), canonicalised majority vote | H100 | 6 |
| E7a | Generic GPT-5 agent (low effort): free tool choice over the same tools, ≤6 tool calls, same image, rules and genus list, same GUARD and one repair | H100 | 1, 2 |
| E8 | Blinded evidence-quality rating (Gemma) of A3v13 and E7a traces, plus a 60-case author audit | H100 | 2 |

## 4. Metrics
- **Accuracy:** the v12 task-aware scorer (`quick_semantic_score`, score ≥ 1 counts as correct). Reported
  separately for D and H, and by question type.
- **Visible-bleaching task (comment 7)** (`coralagent/evaluation.py`):
  - Inclusion: CoralVQA bleaching/health questions that explicitly mention bleaching and are not about
    susceptibility. That gives 412 questions: 66 visible bleaching, 346 no visible bleaching.
  - Mapping: the same rule maps reference and predicted answers for every method.
    - "yes", or a count above 0 → visible bleaching.
    - "no", or 0 → no visible bleaching.
    - Anything else → no verdict.
  - Reported: per-class correct / incorrect / no verdict, recall, specificity, F1, coverage and answered-only
    accuracy.
- **Cost and latency:** cost and calls per question for each arm, from the usage logged on every request.

## 5. Statistics
- **Primary:** A3v13 against strict GPT-5 on H (2,400 questions). Paired accuracy difference with a 95% bootstrap CI
  that resamples whole images (10,000 resamples, seed 2026), plus an exact McNemar test.
- **Secondary (Holm-corrected):**
  - A3v13 against Grok and against Gemma on H.
  - A3v13 against E5, E6 and E7a on H100.
  - Each ablation against A3v13 on D.
- **Reporting:** whatever the outcome, every result is reported.

## 6. Budget and safety rules (enforced in `coralagent/run.py`)
- **Batch APIs only:** OpenAI, xAI and Together batch jobs. No synchronous inference calls.
- **Caps:** OpenAI $38 (of $40), xAI $80, Together $10. The caps can be lowered in `control/CONTROL.json`.
- **Worst-case check:** before every batch, the worst-case cost (input plus the full output allowance) is checked
  against the remaining budget.
- **Test batch first:** 10 requests per provider per stage. Anything larger is sent only if all of them succeed.
- **Batch size:** at most 1,000 requests per batch, and one batch per provider per round.
- **No double submission:** batch IDs are stored before polling. A restart waits for open batches and never
  resubmits them.
- **Retries:** at most one resubmission, and only of failed requests.
- **Stop rule:** a round with more than 5% failures stops the stage.

## Deviations
- **2026-11-14, before any result.** In the E0 pilot the Together batch test (Gemma-4-31B) returned no output for 10 of 10 requests (no cost).
  - Together is now optional: if its test batch fails twice in a stage, that stage continues without it and the report says so.
  - GPT-4.1, a main baseline in the manuscript, was added to E1 and E4a (v12 ReefBank prompt, batch).
  - The E8 evidence-quality judge is GPT-4.1 instead of Gemma. GPT-4.1 is neither the system backbone (GPT-5) nor the verifier (Grok).
- **2026-11-14.** All stages are allowed to run unattended in order. The safety rules in section 6 still stop a stage on any failure.
- **2026-11-15, before any held-out result.** E1 test batch: with v12's strict settings (120 output tokens, default reasoning effort) 9 of 10 GPT-5 answers were empty. Reasoning tokens used up the budget, the same failure as fix 1. v12 masked it with up to three retries.
  - The strict GPT-5 baseline (E1, E4a, E6) now uses the same GPT-5 setting as every other v13 GPT-5 call: reasoning effort "low", 2,000 output tokens. The prompt is unchanged.
  - The prior for the held-out images comes from this run. For images 1–100 (ablations only) the prior stays the valid v12 strict answers.
  - Cost of the failed test batch: $0.008.
- **2026-11-15, before any E3 result.** Measured E0 and E1 costs project the OpenAI total close to the $38 cap. The optional A7 ablation (+chart/segmentation/bleaching tools) is dropped. The reviewer did not request it, and the manuscript already excludes that branch from the proposed system.

- **2026-11-15, during E1, before any result is analysed.** Two operational fixes. (a) OpenAI batch files are now split by model. E1 mixed GPT-5 and GPT-4.1 in one file, OpenAI rejected 358 lines without charge, and those requests used their one resubmission. (b) The 5% stop rule now applies only to rounds of at least 50 requests, or to rounds with 10 or more failures. E1 stopped on a 2-request tail round where both requests (GPT-4.1 outputs that failed to parse twice) failed. Those two answers are kept as failures and scored as wrong. Neither change touches prompts, models or scoring.
- **2026-11-16, before E3–E8 start.** Measured OpenAI costs are about twice the plan's estimates (E2 is ~$0.005 per question). To answer every reviewer comment within the $40 OpenAI budget:
  - (a) The run order is now E2 → E4 → E5 → E6 → E7a → E8 → E3, so the stages behind C1–C4 and C6 run first.
  - (b) The E8 evidence judge is Grok-4.3 instead of GPT-4.1. Both judged systems (A3v13 and E7a) use the same Grok GUARD, so the judge treats them alike, and the 60-case author audit with κ still applies.
  - (c) The size of the E3 ablation set is fixed just before E3 starts, by remaining budget only, never by results. All 100 D images if at least $11 of the OpenAI cap remains; otherwise the 50-image subset `image_subset(range(100), 50, seed=2026)`; otherwise 25 images drawn the same way. A5 (no GUARD) is derived from E2 at no cost and covers all of D and H.
  - Clarification of (c), committed before E3: "otherwise 50" means at least $5.50 left. Below that, 25 images. The rule is applied automatically when E3 starts (`e3_images: "auto"`).
- **2026-11-16, during E4, before any E4 result.** Parser fix. With the genus list in the prompt, GPT-4.1 wrote 207 of 349 answers as JSON with unquoted values (for example `{"answer": Cyphastrea, ...}`). The v12 parser rejected these, which tripped the stop rule. The v13 parser now quotes such bare values when, and only when, the v12 parser fails. Output that parsed before is unaffected. The fix applies to every method and is not scored differently. These answers were already paid for, so no requests were resubmitted. The 2 GPT-4.1 E1 failures stay as they were scored (as wrong).
- **2026-11-17, during E3, before any E3 result was looked at.** E3 extended to all 100 D images. The automatic rule picked 50 images because the worst-case estimate left $10.94. In practice 933 of 1,800 E3 rows were reused from E2 results at no cost, and the first wave is expected to cost about $2.75. The full D set is projected at about $8, which keeps the OpenAI total near $35, under the $38 cap. This is a budget-only decision; no E3 result was inspected.
  - Mechanics: the running E3 finishes on its 50 images. A follow-on stage E3b then runs the same six variants on all 100 D images; rows already done are reused at no cost.
- **2026-11-18, after E8. Added post hoc (Mustafa's decision).** No coral expert is available for the 60-case author audit. Instead, a second independent LLM judge (GPT-4.1, stage E8b) rates the same 1,200 evidence traces with the identical blinded rubric and prompt. Agreement with the Grok judge is reported as Cohen's κ, quadratic-weighted for the ordinal ratings, along with whether the E8 conclusions hold under the second judge. GPT-4.1 is neither system's backbone, and both judged systems share the same Grok GUARD.
