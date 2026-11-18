# E2 headline results (computed 2026-11-16)

Scoring: v12 `quick_semantic_score` (binary on this data). Statistics follow PROTOCOL.md: paired image-cluster bootstrap (10k resamples, seed 2026) and exact McNemar. Script: `analysis/e2_headline.py`.

## Accuracy

| Method | D (600) | H (2,400) |
|---|---|---|
| A3v13 (CoralAgent-TRACE, with GUARD) | 0.503 | 0.520 |
| A5v13 (same run, answer before GUARD) | 0.532 | 0.534 |
| Strict GPT-5 (B0b) | — | 0.500 |
| GPT-4.1 (B1) | — | 0.527 |
| Grok-4.3 (B2) | — | 0.483 |

## Paired comparisons on H (A3v13 minus the other method)

| Comparison | Difference | 95% CI | McNemar b/c | p |
|---|---|---|---|---|
| **Primary:** vs strict GPT-5 | +0.020 | +0.004 to +0.036 | 195/147 | 0.011 |
| vs GPT-4.1 | −0.007 | −0.028 to +0.013 | 276/292 | 0.53 |
| vs Grok-4.3 | +0.038 | +0.015 to +0.060 | 380/289 | 0.0005 |
| vs no GUARD (A5) | −0.013 | −0.021 to −0.006 | 21/53 | 0.0003 |

The secondary p-values still need Holm correction across the full family of tests once E4–E7a are in.

## By question type on H (A3v13)

| Type | Accuracy |
|---|---|
| bleaching_health (555) | 0.759 |
| other (400) | 0.680 |
| color_visual (347) | 0.556 |
| morphology_visual (349) | 0.413 |
| presence_count (400) | 0.318 |
| genus_taxonomy (349) | 0.264 |

For comparison on bleaching_health: strict GPT-5 0.634, GPT-4.1 0.555, Grok 0.506. On genus_taxonomy A3v13 trails strict GPT-5 (0.350) and GPT-4.1 (0.395).

## Reading

- The pre-registered primary test favours CoralAgent-TRACE over strict GPT-5 on held-out images (+2.0 points, CI excludes 0).
- The gain is concentrated in bleaching and health questions, the paper's main target task (+12.5 points over strict GPT-5).
- CoralAgent-TRACE is statistically indistinguishable from GPT-4.1 overall.
- GUARD lowers accuracy: it changed 74 H answers, 21 for the better and 53 for the worse. This was already visible in E0 and must be reported as is (C5). No tuning, per the protocol.
