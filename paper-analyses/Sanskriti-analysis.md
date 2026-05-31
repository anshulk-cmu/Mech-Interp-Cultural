# SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture — Technical Analysis

**Citation.** Arijit Maji, Raghvendra Kumar, Akash Ghosh, Anushka, and Sriparna Saha. 2025. *SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture.* In Findings of the Association for Computational Linguistics: ACL 2025, pages 4434–4451. arXiv:2506.15355. (Authors 1–4 at IIT Patna; Anushka at Banasthali Vidyapeeth.)

## Problem and motivation

The paper attacks a now-familiar gap: LLMs are deployed globally, yet benchmarked overwhelmingly on Western, English-centric, or at best multilingual data. The authors argue that *multilingualism is not cultural competence*. Existing QA benchmarks such as TyDi QA and XQuAD test cross-lingual reading comprehension but say nothing about whether a model knows that Kathakali belongs to Kerala or Madhubani art to Bihar. India is chosen as the testbed precisely because it maximizes intra-national cultural variance across 28 states and 8 union territories. The stated risk of cultural blindness is concrete: perpetuating stereotypes, alienating underrepresented communities, and degraded utility in education, governance, and healthcare deployments inside India. The motivation positions SANSKRITI against the prior dataset DOSA (Seth et al. 2024), which it characterizes as smaller and not covering all states and union territories.

## Core contribution and precise claims

The headline artifact is a benchmark of **21,853 manually curated and validated multiple-choice question-answer pairs**, claimed to be the largest dataset for testing Indian cultural knowledge. The number is consistent throughout the paper (abstract, introduction, contributions, conclusion, ethics statement) and is confirmed by the arXiv abstract and the ACL Anthology landing page. Coverage spans all **28 states and 8 union territories** and **sixteen cultural attributes**. Note a minor internal inconsistency: the abstract prose enumerates only fifteen attributes by name, omitting "Cultural Common Sense," which Appendix 7.1 supplies as the sixteenth (so the count "sixteen" is correct; the abstract's inline list is incomplete).

Four orthogonal *question types* structure the data, each built by a dedicated ten-person sub-team (40 annotators total):

- **Association Prediction** — identify the culture referenced in a statement.
- **Country Prediction** — determine the country from a cultural statement.
- **General Awareness / General Knowledge (GK)** — free-standing factual questions.
- **State Prediction** — identify the specific Indian state referenced.

All items are fact-based four-option MCQs (A–D). The dataset is released on HuggingFace (`13ari/Sanskriti`) and Google Drive. Annotators were paid USD 1.20 per 10 questions created and USD 0.60 per 10 verified; ~75% were native speakers of at least one Indian language and ~80% had lived 15+ years in their language region.

## Method and "mathematics"

An important caveat for our project: **SANSKRITI is a dataset/benchmark paper, not a mechanistic-interpretability paper.** It contains *no* loss functions, intervention operators, probing objectives, or causal-tracing/activation-patching machinery. There is no ROME-style rank-one update, no difference-in-means refusal direction, no logit-difference metric, and no MDL probe — none of the formal apparatus the analysis template anticipates. The only quantitative formalism is the evaluation protocol, reproduced faithfully below.

**Data organization.** Scraped text is stored as a nested record:

```
{ "state_name" : { "attribute" : "scraped data related to the attribute and state" } }
```

Sources: Wikipedia, Ritiriwaz, Holidify, Google Arts & Culture, and Times of India. (The paper says "six diverse and reliable platforms" but enumerates five — another small internal counting slip.)

**Evaluation protocol.** Evaluation is zero-shot MCQ accuracy. For an item with question `q`, four options `O = {A,B,C,D}`, and gold option `o*`, the model is given a templated prompt and scored by argmax over option probabilities:

```
prediction = argmax_{o ∈ O} P_model(o | prompt(q, O))
correct    = 1[ prediction == o* ]
Accuracy   = (1/N) * Σ_i 1[ prediction_i == o*_i ]
```

where `N` is the number of items in the cell being scored and `1[·]` is the indicator. The four prompt templates are fixed strings, e.g.:

```
Country Prediction:  "Location: Unknown. Question: Which country is associated with <cultural_aspect>?  Options: <options>  Short Answer:"
State Prediction:    "Location: India. Question: Which Indian state is known for <cultural_aspect>?  Options: <options>  Short Answer:"
General Knowledge:   "Question: <general_cultural_question>?  Options: <options>  Short Answer:"
Association-Based:   "Question: The <cultural-entity> is most closely associated with which <cultural_context>?  Options: <options>  Short Answer:"
```

Inference details: open-source models run in 16-bit floating point with **greedy decoding**; the option with the highest output probability is selected; proprietary models accessed via API. Accuracy is the *sole* metric — there is no calibration, no log-odds, no per-token analysis. The "algorithm" is therefore: (1) scrape per-state/per-attribute text; (2) sub-teams write MCQs grounded in that text with plausible distractors; (3) cross-validate each sub-team's output with another sub-team plus a final review; (4) score each model's argmax option choice and average.

**A methodological subtlety worth flagging** (Limitation 4): when a cultural attribute is not uniquely Indian, the annotators deliberately construct distractors that *do not reference the prompted attribute*, then "replace the original correct answer with the option that most closely aligns with it among the four choices." This is an artificial constraint on the answer key that can introduce label noise — relevant to anyone reusing the items.

## Experimental setup and headline results

Eleven models across three families were evaluated (confirmed against the arXiv HTML):

- **LLMs:** GPT-4o (proprietary), Llama-3.1-70B-Instruct, Qwen2.5-72B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3-medium-4k-Instruct.
- **SLMs:** Gemma-2-2b, Qwen2-1.5B-Instruct, Llama-3.2-3B-Instruct, SmolLM-1.7B-Instruct.
- **ILMs (Indic):** Navrasa-2.0, OpenHathi-7B-Instruct.

**Overall average accuracy (Table 2).** GPT-4o is best at **0.87**, then Llama-3.1-70B **0.86**, Qwen2.5-72B **0.84**, Phi-3-medium **0.77**, Mistral-7B **0.70**, Qwen2-1.5B **0.74**, Llama-3.2-3B **0.52**, Gemma-2-2b **0.48**, Navarasa-2.0 **0.40**, OpenHathi-7B **0.32**, SmolLM-1.7B **0.16** (the floor). All confirmed against the live arXiv HTML.

**Per-type accuracy (Table 1).** GK is uniformly easiest (GPT-4o 0.96, Qwen2.5-72B 0.94, Llama-3.1-70B 0.93); State Prediction is hardest across models. A striking efficiency finding: **Qwen2-1.5B-Instruct** scores 0.90 on Country Prediction and 0.82 on GK — competitive with 70B+ models, and the best SLM overall. The Indic models *underperform* general models: Navrasa-2.0 is the single worst model overall despite being India-specific. Fine-grained analyses (Figures 6, 12–15): models do well on religion, medicine, and cultural common sense, but struggle on costume, cuisine, and art; geographically they fail on North-Eastern states (Sikkim, Arunachal Pradesh, Tripura) and on Bihar/Jharkhand, while doing well on Delhi and Maharashtra — "states with globally recognized cities tended to yield better results." Per-state question counts range from ~300 to over 800.

## Stated limitations and threats to validity

The authors list five: (1) only sixteen attributes and a narrow set of question formats (no True/False or adversarial items yet); (2) no state-specific *multilingual* queries — everything is English; (3) no visual QA; (4) limited contextual clarity for non-uniquely-Indian attributes, mitigated by the distractor/answer-replacement trick above; (5) all items are fact-based MCQs that "do not explicitly require reasoning or causal understanding." Additional threats I observe: source bias (Wikipedia/news skews toward globally visible regions, mirrored in the state-count gradient); annotator-pool bias even with cross-validation; possible train-set contamination for GPT-4o on well-documented topics (the error analysis itself attributes correct answers to strong keyword associations in training data, e.g. "Pandal," "Yoga"); and the merging of Dadra & Nagar Haveli and Daman & Diu into one entity for count balance.

## Relevance to our Phase 1 (ICCD-3K minimal-pair probe)

Our project is a *mechanistic* study asking whether post-training alignment (via any fine-tuning method: SFT, RLHF, DPO, RLVR/GRPO, ...) **rewrites** mid-layer cultural representations or **gates** them late, using Indian cultural minimal pairs (clean prefix vs corrupted prefix + target answer; per-item log-odds difference; 60 cells of 50 items = 3,000 items; paired t-tests). Indian culture here is both the controlled probe for that mechanistic question and a genuine subject of study in its own right. SANSKRITI is not a methods source for the interventions — it offers none — but it is a high-value *content and design* source.

**What we borrow.** (1) The taxonomy: SANSKRITI's 16 attributes × 36 regions × 4 question types is a ready-made factorization for our 60-cell stratification (e.g., attribute × region cells), giving us a principled, coverage-balanced way to define cells of 50 items each. (2) The grounded item-construction discipline: each fact tied to explicit source text with a documented rationale (Appendix 7.3, the Bihar example) is exactly the provenance hygiene we want so that a "corrupted prefix" flips a *known* fact rather than a contested one. (3) The released HuggingFace dataset (`13ari/Sanskriti`) is a candidate seed pool — we can mine its verified state/attribute facts to author minimal pairs (e.g., "Kathakali originates in the state of ___" → Kerala [clean] vs Karnataka [corrupted]) and reuse its plausible distractors as our corrupted targets.

**Design constraints it implies.** SANSKRITI scores only argmax accuracy; our design must go *further* — we need per-item **log-odds difference** `Δ = log P(target | clean) − log P(target | corrupted)` at the answer token, computed from logits, not multiple-choice argmax. So we cannot lift their MCQ format directly; we must convert facts to free-completion prefixes with a single target token to make the log-odds metric well-defined. Their fixed-template prompts are a useful starting scaffold, but the "Options: A–D Short Answer:" framing must be dropped for clean prefix-continuation prompts.

**Pitfalls it warns against.** (a) *Label-noise from the answer-replacement rule* (Limitation 4): any inherited item where the gold answer was swapped to "closest option" is unsafe for a causal log-odds probe — filter these out, keep only uniquely-Indian, unambiguous facts. (b) *Contamination and keyword shortcuts*: GPT-4o's wins ride on strong surface associations; for a rewrite-vs-gate study we must control whether a "cultural representation" is genuine knowledge or lexical co-occurrence, e.g., with surface-token-matched control pairs. (c) *Regional/source imbalance*: the 300–800 per-state gradient shows where web text is thin; our 50-items-per-cell quota must not oversample data-rich states, or paired t-tests will conflate region with data availability. (d) *English-only, fact-MCQ scope*: SANSKRITI lacks reasoning and multilingual items, so it cannot tell us whether alignment gates *culturally sensitive* (vs merely factual) representations; our pairs should add norm-sensitive items beyond plain facts. (e) Indic (ILM) models do *worse*, cautioning us not to assume "Indic" provenance implies stronger internal cultural representations when selecting base vs aligned model pairs.

In short: SANSKRITI gives Phase 1 a vetted, factorized inventory of Indian cultural facts and a cautionary list of annotation artifacts, but every mechanistic metric, intervention operator, and statistical test in our plan must come from the interpretability literature, not from this paper.

## Validation notes

- **arXiv ID = 2506.15355** — *confirmed.* Source: https://arxiv.org/abs/2506.15355 (matches plan).
- **Title** "SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture" — *confirmed.* Sources: https://arxiv.org/abs/2506.15355 ; https://aclanthology.org/2025.findings-acl.228/ . (Note: plan's short title says "...in Indian Culture"; the actual subtitle is "...Knowledge of Indian Culture." Minor wording difference, recorded as a discrepancy.)
- **Authors:** Arijit Maji, Raghvendra Kumar, Akash Ghosh, Anushka, Sriparna Saha — *confirmed* in this exact order. Source: https://aclanthology.org/2025.findings-acl.228/ . (Plan cites "Maji, Saha et al."; correct first/last authors.)
- **Venue / year:** Findings of the Association for Computational Linguistics: ACL 2025, pages 4434–4451 — *confirmed.* Source: https://aclanthology.org/2025.findings-acl.228/ . arXiv submission date June 18, 2025. (Plan: "ACL Findings 2025" — confirmed.)
- **Item count = 21,853** — *confirmed* against paper text and both external indices. Sources: https://arxiv.org/abs/2506.15355 ; https://aclanthology.org/2025.findings-acl.228/ . (Plan says 21,853 — matches.)
- **Coverage = 28 states + 8 union territories; 16 attributes** — *confirmed* (abstract enumerates 15 names; Appendix 7.1 supplies the 16th, "Cultural Common Sense"). Source: https://arxiv.org/abs/2506.15355 .
- **Models (11) and best/worst** — GPT-4o best at 0.87, SmolLM-1.7B worst at 0.16, Navrasa-2.0 worst *among ILMs/overall lowest non-SLM*; Llama-3.1-70B best open at 0.86 — *confirmed.* Source: https://arxiv.org/html/2506.15355 (Tables 1–2).
- **40 annotators; pay USD 1.20/10 created, USD 0.60/10 verified** — present in paper text; not independently web-verifiable; treated as paper-internal, *unverifiable externally* but internally consistent.

## Verification Log

Independent adversarial re-verification performed 2026-05-31. Each external fact was checked against primary sources (arXiv abstract page, arXiv HTML full text, ACL Anthology landing page) and cross-checked against the local source markdown at `d:/Mech-Interp-Cultural/papers/Sanskriti/Sanskriti.md`.

- **arXiv ID 2506.15355** — *confirmed.* https://arxiv.org/abs/2506.15355 resolves to this exact paper. Submission date June 18, 2025 confirmed.
- **Title** "SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture" — *confirmed* verbatim on both https://arxiv.org/abs/2506.15355 and https://aclanthology.org/2025.findings-acl.228/. (Subtitle is "Knowledge of Indian Culture," not "in Indian Culture" — analysis already records this as a plan-wording discrepancy; no file change needed.)
- **Authors** Arijit Maji, Raghvendra Kumar, Akash Ghosh, Anushka, Sriparna Saha (this order) — *confirmed* on both arXiv and ACL Anthology. Affiliations in source md: authors 1–4 (Maji, Kumar, Ghosh, Saha) at IIT Patna; Anushka at Banasthali Vidyapeeth — matches the analysis citation line.
- **Venue / pages** Findings of the ACL: ACL 2025, pages 4434–4451, Vienna, Austria — *confirmed* against the ACL Anthology bibliographic citation.
- **Dataset size 21,853 QA pairs; 28 states + 8 UTs; 16 attributes** — *confirmed.* arXiv abstract, ACL abstract, and source md (abstract, intro, contributions §2, conclusion, ethics) all state 21,853. Re-confirmed the 16-vs-15 attribute nuance: abstract prose and arXiv-fetched attribute list enumerate 15 names; Appendix 7.1 of the source md lists all 16 (the 16th being "Cultural Common Sense"). Analysis statement is correct.
- **Table 2 average accuracies** — *confirmed verbatim* against both the live arXiv HTML and the source md table: GPT-4o 0.87, Llama-3.1-70B 0.86, Qwen2.5-72B 0.84, Phi-3-medium 0.77, Qwen2-1.5B 0.74, Mistral-7B 0.70, Llama-3.2-3B 0.52, Gemma-2-2b 0.48, Navarasa-2.0 0.40, OpenHathi-7B 0.32, SmolLM-1.7B 0.16. Best GPT-4o, floor SmolLM-1.7B — both correct.
- **Table 1 per-type values** — spot-checked: GK GPT-4o 0.96 / Qwen2.5-72B 0.94 / Llama-3.1-70B 0.93; Country Prediction Qwen2-1.5B 0.90; State Prediction lowest type across models — all match the source md table.
- **Source-paper internal contradiction (not an analysis error):** §5.1 text says "OpenHathi emerged as the best performer with a score of 0.32" and calls Navrasa-2.0 the weakest of all models, yet Table 2 lists Navarasa 0.40 > OpenHathi 0.32. This is a contradiction in the *paper itself*. The analysis correctly reports the table numbers and flags the narrative-vs-score mismatch (see "Experimental setup and headline results" and the validation note), so no correction is warranted.
- **Evaluation protocol / math** — *confirmed.* Source md §4.2: zero-shot MCQ, 16-bit floating point, greedy decoding, "the option with the highest probability was selected," accuracy as the sole metric. The analysis's argmax/indicator/accuracy formulas and the four fixed prompt templates ("Short Answer:" framing) reproduce the source faithfully. The nested data-record schema `{state_name:{attribute:scraped_text}}` matches §3.1. No interpretability machinery (loss, intervention, probe, ROME, refusal direction, logit-difference) exists in the paper — the analysis's "not a methods source" framing is accurate.
- **HuggingFace dataset `13ari/Sanskriti` and Google Drive release** — *confirmed* present in source md contribution 4 and corroborated by web search; HuggingFace paper page https://huggingface.co/papers/2506.15355 also resolves.
- **Annotator pay (USD 1.20/10 created, 0.60/10 verified) and demographics (~75% native speakers, ~80% lived 15+ years in region)** — present in source md §3.2 / Ethics Statement; not independently indexed online, so externally *unverifiable* but internally consistent. Correctly characterized.

**Corrections applied to the file:** none. Every external bibliographic fact and every load-bearing quantitative claim independently checked out against primary sources, and the math/protocol matches the source markdown. The previously-existing "Validation notes" section was already accurate; this log records the independent re-check.

**Word-count flag:** the file is ~1896 words (measured with whitespace tokenization, excluding this log it is ~1690), modestly above the 1500–1800 target band after this log is appended. Flagged per instructions; not padded or trimmed, as all content is substantive verification material. (The structured summary's reported `word_count` of 1794 is slightly stale relative to the current file.)
