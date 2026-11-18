# E4: the genus list (reviewer comment C4), computed 2026-11-16

Script: `analysis/e4_genus.py`. Scoring and statistics are the same as in `E2_headline.md`.

## Accuracy on genus_taxonomy questions

| Method | n | Without list | With list | Difference | 95% CI | McNemar b/c | p |
|---|---|---|---|---|---|---|---|
| Strict GPT-5 (H) | 349 | 0.350 | 0.287 | −0.063 | −0.112 to −0.014 | 29/51 | 0.018 |
| GPT-4.1 (H) | 349 | 0.395 | 0.244 | −0.152 | −0.198 to −0.106 | 13/66 | 1e-09 |
| Grok-4.3 (H) | 349 | 0.315 | 0.215 | −0.100 | −0.149 to −0.052 | 21/56 | 8e-05 |
| CoralAgent-TRACE (H100 genus) | 92 | 0.315 | 0.293 | −0.022 | −0.120 to +0.076 | 9/11 | 0.82 |

"With list" for CoralAgent-TRACE is A3v13; "without list" is A3v13_no_genus_list.

## Why the list hurts: the prompt shows only the first 90 labels

The v12 code (`_known_genus_prompt(max_items=90)`, carried unchanged into v13 and the E4 baselines, as the protocol requires) builds a list of 392 labels from the CoralVQA genus answers, sorts it alphabetically and puts only the **first 90** into the prompt. That cuts the list off at "Cyphastrea".

- Only **74 of 349** (21%) held-out genus answers appear in the 90 labels shown. The full 392-label list covers 316 (91%).
- The most common correct answers never appear in the prompt: Porites (83 questions), Pocillopora (33), Montipora (28), Stylophora (13) and Orbicella (7).
- Told to "choose one exact label", models swap a correct "Porites" for a shown label such as Acropora or Cyphastrea. This is the main pattern in the GPT-4.1 answers the list broke.
- The 392 labels also include non-genus answers from the dataset ("Brown", "Clear water", "Boulder-shaped", "Yes", "No").

## Reading

- As implemented in the published system, the closed genus list hurts every model that receives it (−6 to −15 points on held-out genus questions).
- For CoralAgent-TRACE, removing the list makes no significant difference.
- This is a v12 implementation flaw of the same kind as the others found in the audit. It explains part of the system's weak genus accuracy (0.264 on H).
- Per the protocol, the list is not changed after seeing held-out results. Any run with the full list would be a separately labelled, post-hoc analysis.
