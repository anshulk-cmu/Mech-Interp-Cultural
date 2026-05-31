# INDICA — Technical Analysis

**Paper:** Common to Whom? Regional Cultural Commonsense and LLM Bias in India
**Authors:** Sangmitra Madhusudan, Trush Shashank More, Steph Buongiorno, Renata Dividino, Jad Kabbara, Ali Emami
**Venue / Year:** Accepted to ACL 2026 (Main Conference); arXiv:2601.15550 (submitted 22 Jan 2026, rev. 14 Apr 2026)
**Affiliations:** Emory University; Independent; Brock University; MIT

## Problem and motivation

Cultural commonsense benchmarks have until now treated the nation as the unit of culture: CultureBank (Shi et al., 2024) and CulturalBench (Chiu et al., 2025) ask "what is the Indian answer," implicitly assuming every Indian citizen shares it. INDICA attacks this monolith assumption head-on. India has 28 states, 8 union territories, and 22 official languages, so the authors ask an empirical question that prior work never tested: does cultural commonsense hold uniformly within a nation, or does it vary sub-nationally? The answer matters for evaluation methodology (a single national gold answer may be wrong for most of the country) and for fairness (if a model has one "default" regional answer, it silently erases the others). Existing India work is either factual/exam-style (MILU, SANSKRITI, IndicQuest) or bias-focused (IndiBias, FairI Tales), and treats India as uniform; none measures whether everyday commonsense is regionally partitioned.

## Core contribution and precise claims

INDICA is the first benchmark for *sub-national* cultural-commonsense variation. The concrete deliverables and headline claims are:

1. A human-annotated dataset: 515 questions across 8 OCM-grounded domains, 18 subcategories, and 39 topics, with per-region gold answers yielding **1,630 region-specific question-answer pairs** over five regions (North, South, East, West, Central).
2. The empirical headline: of the 132 questions where all five regions produced a valid answer, only **52 (39.4%)** show unanimous agreement — i.e., cultural commonsense in India is "predominantly regional, not national."
3. Two evaluation gaps across eight frontier LLMs: (a) overall accuracy is only **49.5%–52.6%**, with *fully*-correct rates a low **13.4%–20.9%**; (b) under region-agnostic conditions, all models exhibit geographic bias, over-selecting Central (~1.37× expected) and North (~1.21×) India while under-selecting West (~0.73×) and East (~0.82×).
4. A transferable methodology (OCM-grounded question design → regional collection → consensus gold standard → dual RASA/RA-MCQ evaluation → chi-square bias test), demonstrated on a China template.

## Method and mathematics

### Dataset construction

Question creation is three-staged. Domain selection grounds 8 domains in the **Outline of Cultural Materials** (OCM; Murdock et al., 2008), a 90+ category / 700+ subcategory anthropological taxonomy. Subcategories are filtered by three criteria (diversity, non-overlap, everyday-not-institutional), yielding 18 subcategories. GPT-4-0613 (T=0.7, top-p=1.0) generated 8–10 candidate topics per subcategory; 2–4 were manually kept per criteria, giving 39 topics. From 3–8 manual seed questions per topic, GPT-4-0613 generated ≥15 questions each; manual review against ambiguity/redundancy/cultural-grounding criteria produced **611 unique questions**.

Response collection used **5 participants per region** (25 total), each required to have lived in-region for the majority of life, recruited on Prolific (IRB-approved, $8.00/hr). Each answered all 611 questions in free text, giving 5 × 611 × 5 = **15,275 responses**.

### Gold-standard consensus

The agreement threshold is a piecewise rule (Eq. 1):

```
Threshold(N) = 4                    if N >= 5
             = max(2, N - 1)        otherwise
```

with `N` = number of participant responses for a question in a region. Since every cell had N=5, the operative rule is **4-of-5 (80%) intra-region agreement**. GPT-4o-2024-08-06 (T=0.1) gave preliminary semantic-equivalence classifications; two annotators then manually verified every case using a custom tool that showed GPT-4o's label beside all 5 raw responses. This was a *meta-annotation* (is the label right?), not a subjective cultural judgment, and inter-annotator agreement was **Fleiss' κ = 1.0**. Questions clearing threshold in ≥1 region were retained: **515 of 611 (84.3%)**. Override analysis (Appendix Table 32) is revealing: humans overrode GPT-4o on **28.9%** of inter-regional and **24.5%** of universal cases, almost all *removals* (634/636 and 26/26), i.e., GPT-4o systematically over-identified cross-regional consensus — a direct warning about LLM-judged semantic equivalence.

Three agreement levels are defined: **intra-region** (4/5 within a region); **inter-region** (a strict, exact-match rule across all 10 pairs — "silk" vs "silk and cotton" do *not* agree); and **universal** (all five regions express the same concept; festival names must match exactly, so Pongal ≠ Lohri).

### Evaluation tasks and metrics

**RASA (Region-Anchored Short Answer)** prepends the region ("In South India, …"), producing the 1,630 anchored questions, and elicits free-form generation. Gemini 3.0 Flash (T=0.0) judges each response as Correct (1.0) / Partially Correct (0.5) / Incorrect (0.0). Overall accuracy is:

```
Overall = FullyCorrect + w * PartiallyCorrect,   w = 0.5
```

Robustness is shown by sweeping `w ∈ {0.3, 0.5, 0.7}`; rankings hold and models cluster within 3.1–3.7 pp.

**RA-MCQ (Region-Agnostic MCQ)** strips region labels and forces a choice among options, each mapped (hidden from the model) to a region or region-set; this yields **79 questions**. Each question is run **n=30** times at **T=1.0** with randomized option order. Bias is tested with a chi-square goodness-of-fit against uniform (~20%/region). The credit-splitting accounts for multi-region options and variable option counts (3–5). The observed count (Eq. 2) is

```
O_r = sum over q in Q of  1/|R_selected(q)|   if r in R_selected(q), else 0
```

where `Q` is all 30-run instances and `R_selected(q)` is the region-set of the chosen option. Expected counts under uniform selection (Eqs. 3–4): for an instance with `n_q` options where option `i` covers region-set `R_i`, each region in `R_i` earns `ExpectedCredit_r = (1/n_q)·(1/|R_i|)`, summed to `E_r = Σ_q Σ_{i: r∈R_i} (1/n_q)(1/|R_i|)`. The statistic (Eq. 5) is

```
chi^2 = sum_r (O_r - E_r)^2 / E_r,    df = 4
```

and standardized residuals (Eq. 6) `z_r = (O_r - E_r)/sqrt(E_r)` localize the bias (|z|>1.96, 2.58, 3.29 at α=0.05/0.01/0.001).

## Experimental setup and headline results

**Models (8):** closed — Claude Sonnet 4.5, Gemini 3 Flash, GPT-5.2, Grok-4 Fast; open — DeepSeek-V3.2, Llama 3.3 70B, Mistral Large 3, Qwen3-VL. All at T=1.0, n=30 runs.

**RASA (Table 3, % overall):** Grok-4 Fast 52.6, GPT-5.2 52.4, Claude Sonnet 4.5 51.7, Qwen3-VL 51.6, Gemini 3.0 Flash 51.1, Mistral Large 50.4, Llama 3.3 70B 50.3, DeepSeek-V3.2 49.5. Fully-correct ranges 13.4% (Gemini) to 20.9% (Qwen3-VL); 61.3–75.3% of responses are merely partial. A 100-response audit of Claude's partials found **89% over-explaining** vs 1% under-specifying — models elaborate from a generic template and bury regional precision under "cultural noise." Regional spread is small (3–5 pp fully-correct, North/Central marginally higher); domain spread is larger (Traffic & Transport 20.3–32.4% best; Clothing & Adornment 5–12.9% worst).

**RA-MCQ (Tables 39–40):** every model rejects uniformity at **p<0.001** (χ² from 80.70 for DeepSeek to 209.46 for Qwen3-VL, df=4). Averaged selection: North 23.7% (1.21×), South 18.5% (0.88×), East 15.2% (0.82×), West 15.0% (0.73×), Central 26.9% (1.37×). Central peaks at Gemini 28.8% (1.46×). West has the worst under-selection (residuals −3.1 to −8.2). The authors attribute the North/Central default to three reinforcing causes: Hindi's dominance of Indian-language web text (Kakwani et al., 2020), tokenizer fragmentation of low-resource South/East languages (Rust et al., 2021), and Hindi/Bollywood media centrality.

**Dataset characteristics worth noting:** regional coverage is uneven — West 354 questions (68.7%), Central 348, North/South 326 each, East 276 (53.6%, "greater internal diversity"). Domain universal-agreement varies from Festivals & Rituals 1.8% to Traffic & Transport ~2.6% (Table 2's stated 22.6% in §2.4.3 text conflicts with its own Table 2 value of 2.6% — an internal inconsistency).

## Limitations and threats to validity

The authors flag: (1) **geographic scope** — only India is instantiated; (2) **five-region granularity** aggregates large internal diversity (South India spans Tamil/Telugu/Kannada/Malayalam cultures); (3) **participant skew** — Prolific recruits English-speaking, digitally connected, likely urban Indians, so rural practices may differ; (4) **temporal validity** — a 2025 snapshot (data collected Oct–Nov 2025); (5) **domain coverage** — 8 of many possible OCM domains; (6) **no demographic stratification** by religion, caste, gender. Beyond these, the small **N=5 per region** and reliance on an LLM judge (Gemini 3.0 Flash for RASA, GPT-4o for consensus) are methodological risks the override table partially quantifies.

## Relevance to our project (ICCD-3K, Phase 1)

ICCD-3K builds 3,000 Indian cultural minimal pairs (clean prefix vs corrupted prefix + target), 60 cells of 50 items, with a per-item log-odds difference and paired t-tests, to ask whether RLHF *rewrites* mid-layer cultural representations or *gates* them late. INDICA is one of three source datasets (with SANSKRITI and Wikipedia) and contributes several concrete things to Phase 1.

**What we borrow.** First, the **five-region taxonomy** itself: the plan locks the state→region mapping to INDICA's five regions because INDICA empirically validates them, and uses INDICA's regional consensus to confirm a cultural anchor's regional attribution where it overlaps SANSKRITI (mostly festival items). Second, INDICA's **regional under-representation finding** is exactly the behavioral phenomenon our mechanistic study seeks to explain: it documents *that* models default to Central/North; ICCD-3K asks *where in the network* that default is encoded and whether alignment created or merely surfaced it. The averaged selection ratios (Central 1.37×, West 0.73×) give us a target effect that should appear in mid-layer probes if cultural region is linearly represented.

**Design constraints it implies.** INDICA's strict consensus rules (exact festival-name match; "silk" ≠ "silk and cotton") justify our requirement that each clean anchor be unambiguously tied to one region — partial overlaps should not be used as clean/corrupt contrasts. Its **80% (4/5) intra-region threshold** is a defensible bar for treating an item's regional attribution as gold. Its English-only, urban-skewed sampling matches our own stated English-only and five-region limitations, so the threat profiles are aligned rather than additive.

**Pitfalls it warns against.** The override table is the sharpest lesson: an LLM judge **over-identified cross-regional consensus** (28.9%/24.5% override, almost all removals). For ICCD-3K this means any LLM-assisted Stage-7/8 validation of "is this anchor really region-specific" must be human-audited, and we should expect the LLM to *over*-merge regions. The **over-explaining failure mode** (89% of partials) tells us that free-generation scoring is noisy for cultural specificity; our log-odds-on-target design sidesteps this by scoring a fixed target token sequence rather than judging open text, which is a cleaner mechanistic signal. Finally, INDICA's own §2.4.3 text/Table 2 inconsistency on Traffic & Transport agreement (22.6% vs 2.6%) is a reminder to recompute any borrowed statistic from the released data rather than quoting the prose.

The minimal-pair backbone of ICCD-3K follows CounterFact/ROME (Meng et al., 2022, arXiv:2202.05262, NeurIPS 2022), not INDICA; INDICA supplies the cultural *content and taxonomy*, while the rank-one editing and causal-tracing machinery come from the ROME line and the activation-patching / refusal-direction work the plan also cites.

## Validation notes

- **arXiv id 2601.15550** — *confirmed.* The plan cites arXiv:2601.15550; arXiv lists exactly this id for the paper. Source: https://arxiv.org/abs/2601.15550
- **Title "Common to Whom? Regional Cultural Commonsense and LLM Bias in India"** — *confirmed* against arXiv abstract page. Source: https://arxiv.org/abs/2601.15550
- **Authors (Madhusudan, More, Buongiorno, Dividino, Kabbara, Emami)** — *confirmed*, order matches the markdown and arXiv. Source: https://arxiv.org/abs/2601.15550
- **Venue/year: ACL 2026 Main Conference, 2026 preprint** — *confirmed* via arXiv comments field ("Accepted to ACL 2026 Main Conference"); submitted 22 Jan 2026, revised 14 Apr 2026. The project plan lists only "arXiv:2601.15550" without the ACL 2026 acceptance — minor under-specification, not an error. Source: https://arxiv.org/abs/2601.15550
- **Headline numbers (515 questions, 1,630 QA pairs, 8 domains, 5 regions, 39.4% universal agreement, 13.4–20.9% accuracy, Central/North over-selection)** — *confirmed*: web search summary of the arXiv abstract reproduces all of these exactly. Source: https://arxiv.org/abs/2601.15550
- **Kakwani et al., 2020 IndicNLPSuite (Hindi web-text dominance citation)** — *confirmed* as Findings of EMNLP 2020, authors Kakwani, Kunchukuttan, Golla, Gokul N.C., Bhattacharyya, Khapra, Kumar. *Minor correction:* the corpus is ~8.8B tokens total across 11 languages; INDICA's prose "Hindi has billions of tokens in Common Crawl" is a loose paraphrase of relative resource gaps, not a verbatim figure. Source: https://aclanthology.org/2020.findings-emnlp.445/
- **SANSKRITI (companion source dataset)** — *confirmed*: arXiv:2506.15355, "SANSKRITI: A Comprehensive Benchmark…", 21,853 QA pairs, 16 cultural attributes, ACL Findings 2025. Source: https://arxiv.org/abs/2506.15355
- **CounterFact/ROME (Meng et al., 2022) used in our minimal-pair backbone** — *confirmed*: arXiv:2202.05262, "Locating and Editing Factual Associations in GPT," Meng, Bau, Andonian, Belinkov, NeurIPS 2022; introduces Rank-One Model Editing and the CounterFact dataset. Source: https://arxiv.org/abs/2202.05262
- **Internal inconsistency (Traffic & Transport universal agreement 22.6% in §2.4.3 vs 2.6% in Table 2)** — *unverifiable externally* (resolved only against the paper text); flagged as an in-paper discrepancy. Source: paper markdown §2.4.3 and Table 2.

## Verification Log

Independent adversarial re-verification performed 2026-05-31. Checked the bibliographic facts online and every load-bearing quantitative claim and equation against the source paper at `d:/Mech-Interp-Cultural/papers/Indica/Indica.md`.

**Bibliographic facts (online):**
- arXiv id **2601.15550** — *confirmed* via arXiv listing and two independent web searches. Source: https://arxiv.org/abs/2601.15550
- Title "Common to Whom? Regional Cultural Commonsense and LLM Bias in India" — *confirmed* verbatim (arXiv abstract page).
- Author list (Madhusudan, More, Buongiorno, Dividino, Kabbara, Emami), in order — *confirmed* verbatim against arXiv.
- Affiliations (Emory University; Independent Researcher; Brock University; MIT) — *confirmed* against paper header (line 5).
- Venue: arXiv **Comments** field reads exactly "Accepted to ACL 2026 Main Conference"; submitted 22 Jan 2026 (v1), last revised 14 Apr 2026 (v3) — *confirmed*. The "rev. 14 Apr 2026" in this analysis's header matches v3.

**Quantitative claims (against source paper md):**
- 515 questions / 1,630 QA pairs / 8 domains / 18 subcategories / 39 topics / 611 original questions / 15,275 responses (5×611×5) / 84.3% retained — *confirmed* (paper §2.2–§2.4).
- 39.4% universal agreement = 52 of 132 — *confirmed* (paper §2.4.3, line 135).
- RASA overall 49.5–52.6%, fully-correct 13.4–20.9%, partial 61.3–75.3% — *confirmed* against Table 3 (lines 217–224).
- 89% over-explaining in 100-response Claude audit (1% under, 10% both) — *confirmed* (line 239).
- RA-MCQ chi² range 80.70 (DeepSeek) to 209.46 (Qwen3-VL), df=4, all p<0.001 — *confirmed* against Table 39 (lines 1589, 1596).
- Averaged regional selection North 23.7% (1.21×), South 18.5% (0.88×), East 15.2% (0.82×), West 15.0% (0.73×), Central 26.9% (1.37×); Gemini Central 28.8% (1.46×) — *confirmed* against Table 40 average row (line 1611) and per-model rows.
- West worst under-selection, standardized residuals −3.1 (Grok) to −8.2 (Gemini) — *confirmed* (line 1619).
- Regional coverage West 354 (68.7%), Central 348, North/South 326 each, East 276 (53.6%) — *confirmed* (line 129).
- 8 models, T=1.0, n=30 runs; judges Gemini 3.0 Flash (RASA) / GPT-4o (consensus) — *confirmed* (lines 73, 171, 207, 209).

**Equations (against Appendix A.4.6 and A.9.2):**
- Eq. 1 Threshold(N)=4 if N≥5 else max(2, N−1) — *confirmed* verbatim (lines 997–998).
- Eqs. 2–6 (O_r credit-split; ExpectedCredit_r=(1/n_q)(1/|R_i|); E_r; χ²=Σ(O_r−E_r)²/E_r, df=4; z_r=(O_r−E_r)/√E_r with |z|>1.96/2.58/3.29) — *confirmed* verbatim, symbols and form match (lines 1465–1522).

**External citation checks:**
- Kakwani et al. 2020 IndicNLPSuite (Findings of EMNLP 2020): IndicCorp = ~8.8B tokens across 11 languages + Indian English — *confirmed*; this analysis's correction of the paper's loose "billions of tokens in Common Crawl" prose is accurate. Source: https://aclanthology.org/2020.findings-emnlp.445/, https://indicnlp.ai4bharat.org/corpora/
- CounterFact/ROME (Meng et al. 2022, arXiv:2202.05262, NeurIPS 2022) — id/year previously confirmed; not re-fetched this pass.

**In-paper inconsistency (re-confirmed against source):** §2.4.3 prose gives Traffic & Transport universal agreement as 22.6% (line 137) while Table 2 lists 2.6% (line 144) for the same domain — genuine in-paper discrepancy, correctly flagged.

**Corrections made this pass:** none. Every bibliographic fact, headline number, and equation in the body of this analysis was checked and found accurate; no Edit to the analysis body was required.

**Word-count flag:** This file is ~1,960–2,020 words (≈1,962 excluding code fences; 2,019 raw), which is **above** the 1,500–1,800 target band by roughly 160–220 words. Flagged per instructions; not trimmed, since no filler was found and trimming was not requested. (The upstream structured summary's stated 1,660 word count does not match the actual file length.)
