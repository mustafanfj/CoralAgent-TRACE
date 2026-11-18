# E3 ablations on D (reviewer comment C5), computed 2026-11-18

All 100 D images (600 questions). Every variant is built on A3v13 (v13 fix 4). A5 is the same E2 run read before GUARD. Statistics: paired image-cluster bootstrap and exact McNemar, Holm across the seven ablations. "better/worse" counts the questions where the variant was right and A3v13 wrong, and the reverse. Script: `analysis/e3_ablations.py`.

```
D questions present for A3v13 and every variant: 600 (images: 100)

| Method | Accuracy (n=600) |
|---|---|
| A3v13 full system | 0.503 |
| A0 SCOPE only (no prior, no tools beyond the crop) | 0.458 |
| A1 prior + SCOPE | 0.528 |
| A2 fixed: all tools on every question | 0.515 |
| A4 no retrieval | 0.502 |
| A5 no GUARD (answer before verification) | 0.532 |
| A6 no baseline prior | 0.493 |
| A3 without SCOPE crops | 0.517 |

| Variant minus A3v13 | Difference | 95% CI | variant better/worse | p | Holm p |
|---|---|---|---|---|---|
| A0 SCOPE only (no prior, no tools beyond the crop) | -0.045 | -0.072 to -0.020 | 23/50 | 0.00212 | 0.0127 |
| A1 prior + SCOPE | +0.025 | +0.002 to +0.050 | 30/15 | 0.0357 | 0.178 |
| A2 fixed: all tools on every question | +0.012 | -0.005 to +0.028 | 13/6 | 0.167 | 0.501 |
| A4 no retrieval | -0.002 | -0.007 to +0.003 | 1/2 | 1 | 1 |
| A5 no GUARD (answer before verification) | +0.028 | +0.013 to +0.045 | 18/1 | 7.63e-05 | 0.000534 |
| A6 no baseline prior | -0.010 | -0.032 to +0.012 | 24/30 | 0.497 | 0.993 |
| A3 without SCOPE crops | +0.013 | -0.002 to +0.030 | 14/6 | 0.115 | 0.461 |

By question type (accuracy):
| Method | bleaching_health | color_visual | genus_taxonomy | morphology_visual | other | presence_count |
|---|---|---|---|---|---|---|
| A3v13 full system | 0.701 (134) | 0.544 (90) | 0.318 (88) | 0.420 (88) | 0.670 (100) | 0.270 (100) |
| A0 SCOPE only (no prior, no tools beyond the crop) | 0.552 (134) | 0.500 (90) | 0.295 (88) | 0.432 (88) | 0.710 (100) | 0.210 (100) |
| A1 prior + SCOPE | 0.746 (134) | 0.544 (90) | 0.352 (88) | 0.443 (88) | 0.720 (100) | 0.260 (100) |
| A2 fixed: all tools on every question | 0.664 (134) | 0.544 (90) | 0.318 (88) | 0.500 (88) | 0.720 (100) | 0.270 (100) |
| A4 no retrieval | 0.694 (134) | 0.544 (90) | 0.318 (88) | 0.420 (88) | 0.670 (100) | 0.270 (100) |
| A5 no GUARD (answer before verification) | 0.739 (134) | 0.544 (90) | 0.318 (88) | 0.500 (88) | 0.720 (100) | 0.270 (100) |
| A6 no baseline prior | 0.724 (134) | 0.500 (90) | 0.273 (88) | 0.466 (88) | 0.680 (100) | 0.210 (100) |
| A3 without SCOPE crops | 0.701 (134) | 0.544 (90) | 0.330 (88) | 0.455 (88) | 0.710 (100) | 0.270 (100) |
```

## Reading

- **GUARD lowers accuracy here too:** removing it adds +2.8 points (Holm p < 0.001; 18 answers fixed vs 1 broken). This matches H (+1.3, `E2_headline.md`). It is the clearest component effect, and it goes against the paper's original claim.
- **The baseline prior and routed tools matter together:** SCOPE alone, without the prior or routed tools (A0), is 4.5 points worse (Holm p = 0.013). Most of the loss is on bleaching questions (0.552 vs 0.701).
- **The simplest strong configuration is prior + SCOPE (A1):** 0.528, +2.5 over the full system. It is not significant after Holm, but it means the extra routed tools, retrieval and GUARD add nothing on D.
- **Retrieval has no measurable effect** (A4: −0.2, 3 answers changed).
- **The other components (prior alone, fixed all tools, SCOPE crops) show no significant effect** (all within ±1.3).
- **For the response letter:** report the component contributions honestly. The gains over strict GPT-5 come from combining the zero-shot prior with the focused crop and the question-type routing. Verification and retrieval do not improve accuracy in this benchmark. GUARD's role should be reframed around exposing disagreement (see `E8_judge.md`), not around accuracy.
