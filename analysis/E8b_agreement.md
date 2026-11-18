# E8b: second independent judge and agreement (computed 2026-11-18)

No coral expert was available for the planned 60-case author audit. A second blinded LLM judge, GPT-4.1, rated the same 1,200 evidence traces (CoralAgent-TRACE and the generic GPT-5 agent on the 600 H100 questions) with the identical rubric and prompt. GPT-4.1 is neither system's backbone. This check was added post hoc and is logged in PROTOCOL.md. Script: `analysis/e8b_agreement.py`.

```
Paired questions rated by both judges for both systems: 600; judge-2 invalid ratings: 0

## Inter-judge agreement (1,200 traces)

| Rating | Weighted κ (quadratic) | Exact agreement |
|---|---|---|
| grounded | 0.04 | 0.85 |
| relevant | 0.08 | 0.89 |
| sufficient | 0.49 | 0.68 |
| overall | 0.59 | 0.43 |
| contradicts_answer | 0.55 (unweighted) | 0.84 |

## E8 conclusions under each judge (TRACE vs generic agent; 95% image-cluster CI of the difference, 2,000 resamples)

| Measure | Judge | TRACE | Generic agent | Difference | 95% CI |
|---|---|---|---|---|---|
| mean overall | Grok-4.3 | 3.297 | 3.805 | -0.508 | -0.618 to -0.397 |
| contradiction rate | Grok-4.3 | 0.400 | 0.178 | +0.222 | +0.178 to +0.267 |
| convincing wrong | Grok-4.3 | 0.337 | 0.570 | -0.233 | -0.299 to -0.169 |
| wrong flagged | Grok-4.3 | 0.564 | 0.255 | +0.309 | +0.242 to +0.377 |
| auroc | Grok-4.3 | 0.674 | 0.590 | +0.084 | +0.030 to +0.135 |
| mean overall | GPT-4.1 | 4.152 | 4.503 | -0.352 | -0.443 to -0.260 |
| contradiction rate | GPT-4.1 | 0.210 | 0.070 | +0.140 | +0.110 to +0.172 |
| convincing wrong | GPT-4.1 | 0.568 | 0.809 | -0.241 | -0.306 to -0.181 |
| wrong flagged | GPT-4.1 | 0.341 | 0.104 | +0.237 | +0.186 to +0.292 |
| auroc | GPT-4.1 | 0.639 | 0.536 | +0.103 | +0.059 to +0.148 |
```

## Reading

- **Agreement is moderate where the ratings vary:** overall κ = 0.59, sufficiency κ = 0.49, contradiction flag κ = 0.55 (84% exact agreement).
- **Grounding and relevance:** the judges agree on 85–89% of traces, but κ is near zero. Almost every trace is rated 2 of 2 on both, and κ is uninformative at such a ceiling (the kappa prevalence paradox). Both judges see the evidence as grounded and relevant for both systems.
- **GPT-4.1 is more lenient** than Grok on every scale: higher overall scores and fewer contradiction flags for both systems.
- **Every E8 conclusion replicates under the second judge,** with confidence intervals that exclude zero:
  - TRACE's wrong answers are less often backed by convincing evidence: −24 points (CI −31 to −18) vs −23 with Grok.
  - TRACE's traces flag more of its errors: +24 points (CI +19 to +29) vs +31 with Grok.
  - Its evidence rating separates right from wrong answers better: AUROC +0.10 (CI +0.06 to +0.15) vs +0.08 with Grok.
  - Its overall evidence score is lower, as with Grok: −0.35 (CI −0.44 to −0.26).
- **Cost:** $2.37 of OpenAI batch spend.
