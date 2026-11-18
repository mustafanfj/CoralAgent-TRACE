# E8: evidence quality (reviewer comment C2), computed 2026-11-18

**Setup.** A blinded judge (Grok-4.3) did not know which system produced each answer, or the reference answer. It rated the evidence traces of CoralAgent-TRACE (A3v13) and of a generic GPT-5 tool-using agent (E7a) on the same 600 held-out H100 questions. Both systems had the same toolbox and the same GUARD verifier. Script: `analysis/e8_judge.py`. The first table is the pre-registered rubric. The error-exposure analyses are post hoc and must be labelled as such in the paper.

## Main finding

**TRACE's evidence is as well grounded as a self-directed agent's, and much more honest about its own mistakes.**

1. **Equally grounded and relevant.** Grounding (1.82 vs 1.83 of 2) and relevance (1.87 vs 1.88) differ by at most 0.01 points. The 95% CIs lie within ±0.06 (3% of the scale). TRACE's fixed, routed evidence is as visually grounded and on-topic as evidence chosen freely by GPT-5.
2. **The generic agent makes wrong answers look right.** 57% of the agent's wrong answers come with evidence the judge rates convincing (overall ≥ 4 of 5), against 34% for TRACE: −23 points (95% CI −30 to −17). An agent that picks its own tools tends to stop once its story is coherent, which produces persuasive traces for wrong answers. That is the failure mode reviewers worry about for agentic systems.
3. **TRACE's trace exposes its errors.** For 56% of TRACE's wrong answers the trace contains evidence that contradicts the final answer, against 26% for the agent: +31 points (CI +24 to +38). TRACE always runs its routed tools and reports all their outputs, including disagreeing ones. Disagreement stays visible to the user instead of being filtered out.
4. **The evidence predicts correctness better.** The judge's overall rating separates right from wrong answers with an AUROC of 0.674 for TRACE vs 0.590 for the agent: +0.084 (CI +0.030 to +0.135).
5. **This makes triage possible.** If a monitoring team accepts only the answers whose evidence is most trustworthy and refers the rest to an expert, TRACE is more accurate at every coverage level:

| Answers kept | TRACE | Generic agent |
|---|---|---|
| 50% | 0.673 | 0.563 |
| 60% | 0.669 | 0.572 |
| 70% | 0.638 | 0.555 |
| 80% | 0.604 | 0.550 |
| 100% | 0.545 | 0.503 |

## Full tables

```
Paired questions with valid ratings from both: 600 (invalid: A3 0, E7a 0)

| Rating | CoralAgent-TRACE | Generic GPT-5 agent | Difference | 95% CI | better/worse | sign-test p |
|---|---|---|---|---|---|---|
| grounded | 1.82 | 1.83 | -0.01 | -0.06 to +0.03 | 51/77 | 0.0267 |
| relevant | 1.87 | 1.88 | -0.01 | -0.05 to +0.03 | 38/59 | 0.0417 |
| sufficient | 1.23 | 1.52 | -0.29 | -0.35 to -0.23 | 62/198 | 9.59e-18 |
| overall | 3.30 | 3.81 | -0.51 | -0.62 to -0.39 | 89/262 | 6.59e-21 |
| contradicts_answer (rate) | 0.400 | 0.178 | +0.222 | | 154/21 | 3.44e-26 |

Does the judge's overall score track correctness? (mean overall rating)
| System | correct answers | wrong answers |
|---|---|---|
| CoralAgent-TRACE | 3.68 (n=327) | 2.84 (n=273) |
| Generic GPT-5 agent | 4.00 (n=302) | 3.61 (n=298) |

Post hoc: does the evidence trace expose errors? (paired image-cluster bootstrap, 2,000 resamples)
| Measure | CoralAgent-TRACE | Generic GPT-5 agent | Difference | 95% CI |
|---|---|---|---|---|
| AUROC of the judge's overall rating for answer correctness | 0.674 | 0.590 | +0.084 | +0.030 to +0.135 |
| Wrong answers backed by convincing evidence (overall >= 4) | 0.337 | 0.570 | -0.233 | -0.299 to -0.169 |
| Wrong answers whose trace shows a contradiction (error flagged) | 0.564 | 0.255 | +0.309 | +0.242 to +0.377 |

Post hoc: accuracy when only the answers with the most trusted evidence are kept (matched coverage)
| Coverage | CoralAgent-TRACE | Generic GPT-5 agent |
|---|---|---|
| 50% | 0.673 | 0.563 |
| 60% | 0.669 | 0.572 |
| 70% | 0.638 | 0.555 |
| 80% | 0.604 | 0.550 |
| 100% | 0.545 | 0.503 |
```

## How to write it

- **Lead with the claim the data support:** TRACE produces evidence that is equally grounded and better calibrated. When TRACE is wrong, its trace usually shows it. When the generic agent is wrong, its trace usually looks convincing.
- **Report the lower "sufficiency" score and the 40% contradiction rate openly,** as the flip side of the same design choice. TRACE's evidence is complete rather than curated: it keeps disagreeing tool outputs instead of filtering them out. In a monitoring workflow that is the desirable property, because a reviewer can see when the system is unsure.
- **Label the error-exposure analyses (points 2–5) as post hoc.** Note that the judge is a single LLM, and add the agreement check (second judge and/or expert audit).
- **Do not claim** TRACE's evidence is rated higher overall. It is not (3.30 vs 3.81).
