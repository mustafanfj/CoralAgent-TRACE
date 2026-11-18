# H100 controls: reviewer comments C1, C2 and C6 (computed 2026-11-17)

These are the 600 pre-registered H100 questions (100 held-out images, seed 2026). Scoring uses the v12 `quick_semantic_score`. Statistics: paired image-cluster bootstrap (10k resamples) and exact McNemar, with Holm correction across these four secondary comparisons. Script: `analysis/h100_controls.py`.

- **E5:** one GPT-5 call given the same information (image, SCOPE crop, top-5 retrieved records, rules, genus list).
- **E6:** strict GPT-5 with self-consistency (k=3, majority vote).
- **E7a:** a generic GPT-5 tool-using agent with the same toolbox, GUARD and one repair, at most 6 tool calls.

```
H100 questions: 600

| Method | n | Accuracy |
|---|---|---|
| CoralAgent-TRACE (A3v13) | 600 | 0.545 |
| A3v13 before GUARD (A5) | 600 | 0.550 |
| E5 single call, same information | 600 | 0.495 |
| E6 strict GPT-5 self-consistency (k=3) | 600 | 0.522 |
| E7a generic GPT-5 agent | 600 | 0.503 |
| Strict GPT-5 (B0b) | 600 | 0.520 |

| A3v13 minus | n | Difference | 95% CI | McNemar b/c | p | Holm p |
|---|---|---|---|---|---|---|
| E5 single call, same information | 600 | +0.050 | +0.022 to +0.078 | 61/31 | 0.00231 | 0.00926 |
| E6 strict GPT-5 self-consistency (k=3) | 600 | +0.023 | -0.005 to +0.053 | 52/38 | 0.17 | 0.229 |
| E7a generic GPT-5 agent | 600 | +0.042 | +0.013 to +0.070 | 51/26 | 0.00587 | 0.0176 |
| Strict GPT-5 (B0b) | 600 | +0.025 | -0.002 to +0.052 | 47/32 | 0.115 | 0.229 |

By question type (accuracy):
| Method | bleaching_health | color_visual | genus_taxonomy | morphology_visual | other | presence_count |
|---|---|---|---|---|---|---|
| A3v13 | 0.775 | 0.568 | 0.293 | 0.462 | 0.780 | 0.300 |
| E5 single call, same information | 0.690 | 0.545 | 0.272 | 0.429 | 0.780 | 0.180 |
| E6 strict GPT-5 self-consistency (k=3) | 0.643 | 0.591 | 0.359 | 0.396 | 0.770 | 0.320 |
| E7a generic GPT-5 agent | 0.744 | 0.557 | 0.250 | 0.429 | 0.740 | 0.210 |
| Strict GPT-5 (B0b) | 0.690 | 0.557 | 0.370 | 0.385 | 0.750 | 0.300 |
```

## Reading

- **C1/C2 (does the structure matter?):** CoralAgent-TRACE beats a generic GPT-5 agent that has the same tools and the same GUARD by +4.2 points (CI +1.3 to +7.0, Holm p = 0.018). The deterministic routing and SCOPE pipeline add accuracy beyond simply giving GPT-5 the tools.
- **C6 (is it just extra information or extra calls?):**
  - A single GPT-5 call with the same information is 5.0 points worse (Holm p = 0.009). Putting the evidence into one prompt does not reproduce the result.
  - A self-consistency control with a matched call budget is 2.3 points behind, but the difference is not significant (Holm p = 0.23). We cannot claim TRACE beats it overall.
- **Where the gain comes from:** bleaching_health (0.775 vs 0.643–0.744 for the controls) and presence_count (vs E5 and E7a).
  - Self-consistency and strict GPT-5 do better on genus questions. Part of that is the truncated genus list (see `E4_genus.md`).
- **Effect sizes:** on H100 the gap over strict GPT-5 is +2.5 (n.s. on 600 questions). On the full H set it is +2.0 (significant on 2,400 questions; `E2_headline.md`).
