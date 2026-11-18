# Remaining analyses (no API calls)

## C4. Evaluation data and identifiers

- Questions evaluated: 3000 on 500 images. D has 600 questions and H has 2400; H100 has 600.
- Source: `CoralVQA_train.jsonl`, the LLaVA-format **training** split of CoralVQA (Hugging Face `CoralReefData/CoralVQA`). No test or cross-region data were used.
- Question types (v12 classifier): bleaching_health 689, color_visual 437, genus_taxonomy 437, morphology_visual 437, other 500, presence_count 500
- All identifiers are in `analysis/evaluation_ids.csv`.

## C7. Visible-bleaching labels

- excluded: bleaching susceptibility or tendency: 249 questions
- excluded: other health (disease, mortality, condition, colour): 28 questions
- included: asks about visible bleaching: 412 questions
- Reference labels in the task: visible bleaching 66, no visible bleaching 346, unmappable 0.
- v12 used one broad 'healthy/bleached' rule over every question type. Any answer containing white, pale or stressed counted as bleached; dead counted as bleached; susceptibility questions were included. That rule produced 776 binary questions, against 689 bleaching_health questions. The v13 task keeps only questions that ask whether bleaching is visible.

Examples (seed 2026):

*excluded: bleaching susceptibility or tendency*
  - "Is it possible to spot the Pocillopora corals, which are susceptible to bleaching, in the image?" → reference "Yes"
  - "Is it possible to identify the bleaching-susceptible Acropora corals in the photograph?" → reference "Yes"
  - "Is it possible to spot the Pocillopora corals, which are susceptible to bleaching, in the image?" → reference "Yes"
*excluded: other health (disease, mortality, condition, colour)*
  - "Does the image appear in black and white or in color?" → reference "Black and white"
  - "Is the coral in the top-center showing signs of health deterioration?" → reference "No"
  - "Can we confirm the health status of the Porites coral genus situated in the center of the image?" → reference "No"
*included: asks about visible bleaching*
  - "Are there any bleached corals visible in the image?" → reference "No"
  - "Can the bleaching of Acropora coral be observed in the image?" → reference "No Acropora coral prone to bleaching can be seen in the image."
  - "Is the image a coral bleaching image?" → reference "No"

## C8. Visible bleaching: per-class outcomes (correct / wrong / no verdict)

**H (held-out, 400 images)**

| Method | Visible bleaching: correct / wrong / no verdict | No visible bleaching: correct / wrong / no verdict | Sensitivity | Specificity | F1 (visible) | Coverage | Answered-only accuracy | Referred (no verdict) |
|---|---|---|---|---|---|---|---|---|
| CoralAgent-TRACE | 17 / 34 / 0 | 248 / 39 / 0 | 0.333 | 0.864 | 0.318 | 1.000 | 0.784 | 0 |
| TRACE before GUARD | 13 / 38 / 0 | 274 / 13 / 0 | 0.255 | 0.955 | 0.338 | 1.000 | 0.849 | 0 |
| Strict GPT-5 | 33 / 18 / 0 | 156 / 131 / 0 | 0.647 | 0.544 | 0.307 | 1.000 | 0.559 | 0 |
| GPT-4.1 | 32 / 16 / 3 | 118 / 150 / 19 | 0.627 | 0.411 | 0.275 | 0.935 | 0.475 | 22 |
| Grok-4.3 | 48 / 3 / 0 | 94 / 191 / 2 | 0.941 | 0.328 | 0.331 | 0.994 | 0.423 | 2 |

**D (development, 100 images)**

| Method | Visible bleaching: correct / wrong / no verdict | No visible bleaching: correct / wrong / no verdict | Sensitivity | Specificity | F1 (visible) | Coverage | Answered-only accuracy | Referred (no verdict) |
|---|---|---|---|---|---|---|---|---|
| CoralAgent-TRACE | 3 / 12 / 0 | 49 / 10 / 0 | 0.200 | 0.831 | 0.214 | 1.000 | 0.703 | 0 |
| TRACE before GUARD | 2 / 13 / 0 | 54 / 5 / 0 | 0.133 | 0.915 | 0.182 | 1.000 | 0.757 | 0 |

**H100 (matched controls)**

| Method | Visible bleaching: correct / wrong / no verdict | No visible bleaching: correct / wrong / no verdict | Sensitivity | Specificity | F1 (visible) | Coverage | Answered-only accuracy | Referred (no verdict) |
|---|---|---|---|---|---|---|---|---|
| CoralAgent-TRACE | 2 / 6 / 0 | 56 / 8 / 0 | 0.250 | 0.875 | 0.222 | 1.000 | 0.806 | 0 |
| Generic GPT-5 agent | 2 / 6 / 0 | 55 / 7 / 2 | 0.250 | 0.859 | 0.235 | 0.972 | 0.814 | 2 |
| Single call, same info | 2 / 4 / 2 | 48 / 9 / 7 | 0.250 | 0.750 | 0.211 | 0.875 | 0.794 | 9 |
| Self-consistency k=3 | 4 / 4 / 0 | 38 / 26 / 0 | 0.500 | 0.594 | 0.211 | 1.000 | 0.583 | 0 |
| Strict GPT-5 | 5 / 3 / 0 | 39 / 25 / 0 | 0.625 | 0.609 | 0.263 | 1.000 | 0.611 | 0 |

No-verdict cases are answers the mapping cannot read as yes/no or a count; they are not technical failures. In v13, TRACE had 0 technical failures (API or parse errors) and 0 empty answers across 3000 questions. In v12, 444 of 3,000 answers were empty.

## C5. What GUARD does (A3v13, all 3,000 questions)

| Split | Question type | n | Verifier flags a problem | Repair attempted | Repair blocked by protection | Answer changed | Changed: wrong→right | Changed: right→wrong |
|---|---|---|---|---|---|---|---|---|
| D | bleaching_health | 134 | 30 | 30 | 0 | 10 | 1 | 6 |
| D | color_visual | 90 | 36 | 36 | 36 | 0 | 0 | 0 |
| D | genus_taxonomy | 88 | 26 | 26 | 26 | 0 | 0 | 0 |
| D | morphology_visual | 88 | 16 | 16 | 0 | 16 | 0 | 7 |
| D | other | 100 | 11 | 11 | 0 | 9 | 0 | 5 |
| D | presence_count | 100 | 56 | 56 | 56 | 0 | 0 | 0 |
| D | all | 600 | 175 | 175 | 118 | 35 | 1 | 18 |
| H | bleaching_health | 555 | 112 | 112 | 0 | 49 | 11 | 30 |
| H | color_visual | 347 | 133 | 133 | 133 | 0 | 0 | 0 |
| H | genus_taxonomy | 349 | 132 | 132 | 128 | 3 | 0 | 2 |
| H | morphology_visual | 349 | 45 | 45 | 0 | 40 | 4 | 14 |
| H | other | 400 | 44 | 44 | 0 | 34 | 6 | 7 |
| H | presence_count | 400 | 245 | 245 | 243 | 2 | 0 | 0 |
| H | all | 2400 | 711 | 711 | 504 | 128 | 21 | 53 |

All other settings are identical: A5 is the same run read before verification, so its tools, prior and pre-verification answer match A3 exactly.

Final verifier verdict (after any repair and re-verification): supported: 2161, unsupported: 626, partially_supported: 213. 'Repair blocked by protection' means the verifier asked for a change but the existing answer was a valid answer of the requested type (genus name, colour, count, yes/no), so v12 protection kept it. Protected answers therefore change only when the pre-verification answer was empty or invalid for the question type.

Examples of GUARD changes on H (seed 2026):

- [morphology_visual] "Can you describe the morphological structure of the coral genus located in the bottom-right corner of the image?" | reference "Branching" | before GUARD "Branching" → after "no coral" (wrong). Verifier: supported
- [other] "Is the water quality around the coral in the center of the image?" | reference "Good" | before GUARD "Clear" → after "No" (wrong). Verifier: focused_vqa returned 'unknown' with rationale 'water clarity cannot be assessed'; agent answered 'No' instead of 'unknown'
- [bleaching_health] "Can the presence of bleaching in Acropora coral be observed in the image?" | reference "No Acropora coral prone to bleaching can be seen in the image." | before GUARD "No" → after "Yes" (wrong). Verifier: answer 'Yes' conflicts with health_status='healthy'; focused_vqa tool returned 'Unclear' with explicit uncertainty on bleaching vs. dead skeleton
- [morphology_visual] "Is the coral genus located in the bottom-center characterized by a dome shape?" | reference "Yes" | before GUARD "Yes" → after "No" (wrong). Verifier: supported
- [bleaching_health] "Does the image contain bleached corals?" | reference "No" | before GUARD "No" → after "Yes" (wrong). Verifier: answer='Yes' conflicts with health_status='healthy' and evidence 'consensus threshold not met'; focused_vqa tool returned 'Unclear' with dead-skeleton uncertainty
- [other] "Can you provide an update on the growth status of the coral located in the center of the image?" | reference "Healthy" | before GUARD "stable" → after "No" (wrong). Verifier: baseline_prior states 'growing'; focused_vqa states 'stable' with no new tips

Tool calls per question (mean; spatial_focus, baseline_prior, retrieval and every model-based tool each count as one call; GUARD and repair are not tool calls):

| Configuration (D) | Mean tool calls |
|---|---|
| A3v13 full system | 3.74 |
| A0 SCOPE only | 2.00 |
| A1 prior + SCOPE | 3.00 |
| A2 fixed all tools | 9.00 |
| A4 no retrieval | 3.52 |
| A6 no prior | 2.74 |
| A3 without SCOPE | 3.74 |

## C10. Routing: question type versus keywords

- Colour or morphology questions that mention 'genus': 769.
- v13 (answer-type priority, Eq. 5) sends 0 of them to genus_identification.
- The original v12 keyword routing sends 213 of them to genus_identification (the cause of Figure 9c/9d).
- v13 accuracy on these questions: 0.460.

## C13. Failure cases (selected at random with seed 2026 from each category, H split)

- missed visible bleaching: 34 cases on H
- wrong genus: 257 cases on H
- wrong answer accepted by GUARD: 737 cases on H
- GUARD changed a correct answer to a wrong one: 53 cases on H
- verifier error contained by protection: 121 cases on H

| Category | Question | Reference | Prior / focused VQA | Before GUARD → final | Verifier | Likely stage |
|---|---|---|---|---|---|---|
| missed visible bleaching | Does the image contain bleached corals? | Yes | No / No | No → No | supported | focused VQA perception |
| missed visible bleaching | Are there bleached corals visible in the image? | Yes | yes / No | No → No | supported | focused VQA perception |
| wrong genus | Which coral genus is located in the top right corner of the image? | Pocillopora | Acropora / Acropora | Acropora → Acropora | supported | focused VQA perception |
| wrong genus | In the image, which coral genus can be found in the top-right corner? | Lobophyllia | Porites / Colpophyllia | Colpophyllia → Colpophyllia | unsupported | focused VQA perception |
| wrong answer accepted by GUARD | What is the morphological structure of the coral genus in the lower left of the image? | Branched | branching / branching | Branching → Branching | supported | focused VQA perception |
| wrong answer accepted by GUARD | Which coral genus is located in the lower left of the image? | Porites | Acropora / Porites | Astreopora → Astreopora | partially_supported | synthesis / tools disagree |
| GUARD changed a correct answer to a wrong one | What is the morphological structure of the coral in the left-bottom area? | Massive | massive / massive | Massive → Encrusting | partially_supported | GUARD repair |
| GUARD changed a correct answer to a wrong one | Is the coral genus located in the bottom-center characterized by a dome shape? | Yes | Yes / no | Yes → No | supported | GUARD repair |
| verifier error contained by protection | Which coral genus is located in the upper center of the image? | Porites | Porites / unclear | Porites → Porites | unsupported | verifier false alarm (no effect) |
| verifier error contained by protection | Which coral genus is located in the left-bottom position? | Porites | Porites / unclear | Porites → Porites | unsupported | verifier false alarm (no effect) |

Full records are in `analysis/failure_cases.json`.

## Cost and model calls per question (batch prices, from logged usage)

| Method | Model calls per question | Cost per question (USD) |
|---|---|---|
| Strict GPT-5 | 1.00 | 0.0011 |
| GPT-4.1 | 1.00 | 0.0011 |
| Grok-4.3 | 1.00 | 0.0026 |
| CoralAgent-TRACE (E2, plus the strict GPT-5 prior it reuses) | 3.93 | 0.0095 |
| Single call, same information (E5) | 1.00 | 0.0020 |
| Self-consistency k=3 (E6; 2 new samples + the strict answer) | 3.00 | 0.0034 |
| Generic GPT-5 agent with GUARD (E7a) | 4.23 | 0.0109 |

On H100: TRACE makes 3.75 tool calls per question in one planned pass. The generic agent makes 2.47 tool calls over 2.09 sequential planning turns.
Latency: all runs used provider batch APIs, so wall-clock times reflect batch queues, not inference. Sequential model rounds per question indicate the latency in interactive use. TRACE needs one parallel tool round, then GUARD and at most one repair and re-verification. The generic agent needs one round per planning turn, then GUARD.
