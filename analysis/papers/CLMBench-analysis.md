# CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing — Technical Analysis

**Paper:** Yucheng Hu, Wei Zhou, Juesi Xiao (Tianjin University). arXiv:2601.17397, submitted 24 January 2026, cs.CL; comments indicate an EACL MME workshop paper. Verified against arXiv (see Validation notes).

## Problem and motivation

Knowledge Editing (KE) updates individual facts inside a trained LLM without full retraining, using locate-and-edit methods such as ROME (Meng et al., 2022a) and mass-editing extensions (MEMIT). The paper attacks a specific weakness in **Multilingual Knowledge Editing (MKE)**: existing benchmarks (ZsRE, Bi-ZsRE) are built by mechanically translating English-centric datasets into the target language. This injects "translationese" artifacts, under-represents culturally native entities (Chinese idioms, local celebrities), and therefore mis-measures what the model actually knows. The authors argue this obscures the real cross-lingual failure modes of editing.

Two gaps motivate the work: (1) no culturally native bilingual editing benchmark exists, and (2) the *mechanism* of cross-lingual edit transfer failure has never been geometrically characterized. The contribution is both a dataset and a mechanistic explanation.

## Core contribution and precise claims

1. **CLM-Bench**, a Chinese-first CounterFact-style benchmark of **1,010** high-quality bilingual pairs spanning 24 knowledge domains (History, Literature, Science, etc.), generated natively in Chinese with DeepSeek-R1 and then aligned to English, rather than translated from English.
2. A **systematic evaluation** of three state-of-the-art batch locate-and-edit methods (MEMIT, AlphaEdit, PMET) across four models, showing **cross-lingual misalignment**: an edit installed in one language largely fails to propagate to the other, and the two behave as independent channels.
3. A **geometric interpretation**: Chinese and English per-layer edit deltas are **nearly orthogonal** (disjoint subspaces), while a mixed-language edit is almost exactly the **linear sum** of the two monolingual edits, i.e. δ_mix ≈ δ_zh + δ_en. This challenges the "interlingua" hypothesis for parameter editing.

## Method and mathematics

### Editing operators being evaluated

The benchmark probes locate-and-edit methods. The canonical operator is ROME's closed-form **rank-one update** of an MLP down-projection weight W (reproduced from Meng et al., 2022a, arXiv:2202.05262; the CLM-Bench paper invokes but does not restate it):

```
minimize  || Ŵ K − V ||   s.t.  Ŵ k* = v*
closed form:
   Ŵ = W + Λ (C^{-1} k*)^T
   Λ = (v* − W k*) / ( (C^{-1} k*)^T k* )
```

Symbols: `W` is the MLP value/down-projection matrix treated as a linear associative memory; `k*` is the key vector (the layer input at the subject's last token); `v*` is the target value optimized so the model emits `target_new`; `C = K Kᵀ = E[k kᵀ]` is the uncentered second-moment (covariance) of keys estimated over a large corpus, encoding existing memories to be preserved; `Λ` is a rank-one coefficient vector. The update is rank one because it adds the outer product `Λ (C^{-1}k*)ᵀ`.

**MEMIT** generalizes this to many facts across a band of mid layers by solving a least-squares system that spreads the residual `(v* − W k*)` over layers; **AlphaEdit** adds a **null-space projection**: updates ΔW are projected into the null space of the preserved-knowledge key matrix (`ΔW · K_preserved ≈ 0`) to cut interference during sequential edits; **PMET** refines the Transformer parameter transformation for more precise edits. All are run through the EasyEdit framework (Wang et al., 2023).

### The four geometric metrics (the paper's own mathematics)

Given two edit-delta matrices Δ₁, Δ₂ ∈ ℝ^{m×n} (e.g. the `mlp_down` projection delta for Chinese vs English at layer 12), the paper measures their relationship four ways.

**Relative error** (tests linear additivity of mixed editing):

```
RelErr = || Δ_mix − (Δ₁ + Δ₂) ||_F  /  || Δ_mix ||_F
```

`||·||_F` is the Frobenius norm; small RelErr means the mixed edit is reconstructed by summing the monolingual edits.

**Neuron-set overlap** via the Jaccard index over the top-k neurons (largest absolute change) in each delta:

```
J = |A ∩ B| / |A ∪ B|
```

`A`, `B` are the top-k neuron index sets of Δ₁, Δ₂. Low J = disjoint neurons; high J = shared neurons.

**Cosine similarity** of the flattened deltas:

```
CosSim = (Δ₁ · Δ₂) / ( ||Δ₁||₂ ||Δ₂||₂ )
```

**Subspace (principal) angle.** SVD each delta `Δ₁ = U₁Σ₁V₁ᵀ`, `Δ₂ = U₂Σ₂V₂ᵀ`; take the top-`k` left singular vectors to span S₁, S₂; the principal angles θᵢ satisfy

```
cos θ_i = max_{u∈S₁} max_{v∈S₂}  uᵀ v ,   i = 1…k       (k = 10)
```

Large θ = orthogonal subspaces.

### Layer selection procedure

To choose the analysis layer, the authors extract hidden states from 1,000 Chinese–English sentence pairs from Bi-ZsRE, take the **final-token hidden state of the subject span**, and build a per-layer "language vector" across layers 9–12. Language separation (projection difference, PCA/t-SNE separability) grows monotonically with depth, peaking at **layer 12** (projection difference > 2.0; raw projections ≈ −2.0 for Chinese vs ≈ 0.2 for English). All delta-based analysis is therefore done on the layer-12 `mlp_down` deltas of Llama3-8B (MEMIT), for batch sizes 1, 10, 100, 1000.

### Evaluation metrics for editing quality

Eight metrics in four families, computed in each language plus a transfer score:
- **Reliability** — accuracy of the edited fact (predicted token vs `target_new`).
- **Generality** — accuracy on `rephrase_prompt` paraphrases.
- **Locality** — preservation of unrelated knowledge (response similarity pre/post edit on `loc`/`loc_ans` control items drawn from ZsRE).
- **zh_score / en_score** — mean of reliability, generality, locality in each language.
- **trans** — cross-lingual transfer: alignment between Chinese-applied and English-applied accuracy.

## Experimental setup and headline numbers

**Models (4):** Llama2-7B-Chinese and Qwen2-7B (Chinese-heavy), Llama3-8B and Mistral-7B (English-dominant). **Methods (3):** MEMIT, AlphaEdit, PMET. **Protocol:** lifelong/sequential batch editing; primary batch = 1,000 monolingual edits per language; a **mixed-lingual** setup with 2,000 bilingual edit triplets (each fact injected with both a Chinese and an English trigger). Ablations on Llama3-8B (MEMIT) over batch sizes {1,10,100,1000} and over different MLP layers; layer ablation repeated on Qwen2-7B.

**Misalignment, headline case (Llama3-8B, MEMIT, batch 1000):** Chinese editing gives zh_score **61.71%** with zh-Reliability **63.21%**, but en-Reliability collapses to **19.55%**; English editing gives en_score **68.31%** with en-Reliability **75.47%** while zh-Reliability is only **44.42%**. The `trans` metric is low for both monolingual directions (30.93% zh, 58.86% en), confirming poor cross-lingual propagation.

**Linear additivity across batch size (Llama3-8B, MEMIT):** monolingual Chinese reliability (Table 2 zh-Reliability) runs 80.00% (b=1), 73.00% (b=10), 74.41% (b=100), 63.21% (b=1000); the corresponding **mixed** Chinese reliability is 100%, 94.17%, 77.20%, 63.21% — nearly identical, demonstrating that the mixed edit inherits each language's monolingual behavior rather than synergizing. (The paper's own prose in Sec. 5.2 mis-lists the b=100 value as "82.83%"; the table value is 74.41%.)

**Catastrophic forgetting at scale:** at batch 1000, AlphaEdit and PMET can collapse — AlphaEdit on Llama3-8B drops to zh_score 1.84% / en_score 4.04% (Chinese editing); generation metrics fall below 20% across several conditions. So algorithm choice, not just layer, governs multilingual viability.

**Geometric results (layer 12, Llama3-8B):**
- Chinese vs English edits: CosSim **0.027–0.042** (near zero); Jaccard **< 0.32** even at batch 1000; subspace angle **> 60°**, up to ≈ **80°** for small batches; RelErr **> 1.40**. → near-orthogonal, disjoint neuron sets.
- Mixed vs (zh+en): CosSim **> 0.976**; top-5% Jaccard **> 0.75**; RelErr **< 0.23**; subspace angle shrinks from **63°** (b=1) to **12°** (b=1000). → δ_mix ≈ δ_zh + δ_en.
- Language bias vs edit selectivity correlation stays near zero (−0.092 to 0.042): misalignment is **not** explained by simple language bias.

## Stated limitations and threats to validity

The authors restrict to Chinese–English only; generalization to other scripts is open. They evaluate only batch methods within the locate-and-edit paradigm (not MEND/SERAC). The mechanistic analysis is confined to **a single layer (12) of one method (MEMIT) on one model (Llama3-8B)**, a partial view. Implicit threats: the dataset is small (1,010 items) and DeepSeek-R1-generated (model bias in fact selection and the contrastive `target_new`); locality uses response similarity rather than a calibrated control; and the geometric section's figure references are internally inconsistent (text cites "Figure 1" for layer-12 metrics the captions place in Figure 3).

## Relevance to our project (ICCD-3K, Phase 1)

Our study asks whether RLHF/alignment **rewrites** mid-layer cultural representations or **gates** them late, using Indian cultural minimal pairs; Phase 1 builds a 3,000-item minimal-pair probe set (clean vs corrupted prefix + target answer; per-item log-odds difference; 60 cells of 50 items; paired t-tests). CLM-Bench is the closest methodological precedent and informs us concretely:

**Methods/metrics we borrow.**
- The **CounterFact minimal-pair schema** maps onto our clean/corrupted design: `prompt + target_new` (counterfactual) vs `ground_truth` (clean) is our corrupted vs clean prefix. Adopting their five fields (prompt, target_new, ground_truth, subject, rephrase_prompt) gives each item a built-in paraphrase (generality) check.
- Their per-token reliability accuracy is the discrete analogue of our per-item log-odds difference; we keep the continuous metric log P(target|clean) − log P(target|corrupt) for statistical power but can report their reliability/generality/locality triad for comparability.
- The **layer-selection procedure** (final-token-of-subject hidden state, sweep layers, pick where representations separate maximally) is exactly the diagnostic for deciding which mid layers to probe for "rewrite vs gate." Their finding that **cultural/language separation grows monotonically with depth, peaking near the upper-middle layers (≈ layer 12 of ~32)** is a strong prior: our Indian-vs-generic separation likely lives in upper-mid layers, so we should center the 60-cell layer grid there.
- The four delta metrics (subspace angle, RelErr, Jaccard, CosSim) port directly to comparing pre- vs post-RLHF representation deltas: **rewrite** predicts large subspace angle / low cosine between base and aligned cultural directions; **gating** predicts mid-layer directions are preserved with change concentrated late.

**Design constraints it implies.**
- **Native-first, not translated.** Their central thesis is that translationese corrupts cultural benchmarks. ICCD-3K must author Indian items natively and avoid back-translation from English templates, or the "cultural" signal becomes a translation artifact.
- **Balanced cells.** Their domain skew (10.1% vs 0.8%) is a cautionary tale; our 60 cells × 50 items must be genuinely balanced to keep paired t-tests well-powered and comparable.
- **Controlled locality items.** Borrow their ZsRE-style `loc`/`loc_ans` neighbors so each cultural pair ships with an unrelated control, separating cultural-representation change from generic capability drift.

**Pitfalls it warns against.**
- **Single-layer / single-model overreach.** Their own limitation — one layer, one method, one model — is the trap our layer/model sweep exists to avoid before claiming "rewrite vs gate."
- **The orthogonality claim is narrow.** They show Chinese and English *edit deltas* are orthogonal, but this concerns gradient-based parameter updates, not where the underlying knowledge lives; we must test activation geometry directly rather than assume linear separability.
- **Generation collapse ≠ knowledge erasure.** Edited models can recall a fact yet fail to generate it in the other language — reliability and generation diverge. So a null log-odds shift post-RLHF could reflect gating of *expression*, not absence of representation; we need both behavioral and representational read-outs to disambiguate.
- **Generator bias.** DeepSeek-R1 authored their facts; any LLM-drafted ICCD-3K items can leak model priors into the corrupted `target_new`, so human bilingual review of every item is non-negotiable.

## Validation notes

- **arXiv id 2601.17397** — *confirmed*; resolves to this exact paper and matches the plan's cited id. https://arxiv.org/abs/2601.17397
- **Title, authors (Yucheng Hu, Wei Zhou, Juesi Xiao; Tianjin University)** — *confirmed*. https://arxiv.org/abs/2601.17397
- **Venue/year** EACL MME workshop; submitted 24 Jan 2026 (cs.CL) — *confirmed* via arXiv; workshop tag from arXiv comments, soft until proceedings appear. https://arxiv.org/abs/2601.17397
- **Dataset 1,010 pairs / 24 domains; models Llama3-8B, Mistral-7B, Qwen2-7B, Llama2-7B-Chinese; methods MEMIT/AlphaEdit/PMET** — *confirmed* from paper Secs. 3.2, 4.1 and abstract. https://arxiv.org/abs/2601.17397
- **Headline metrics** (61.71% / 19.55%, cosine 0.027–0.042, Jaccard <0.32, subspace >60°, RelErr <0.23) — match Tables 1–3 / Sec. 6.2 exactly; too recent for independent web cross-check: *unverifiable* externally, confirmed against the md.
- **ROME rank-one update equation** — *confirmed* as the canonical form from Meng et al., 2022 (arXiv:2202.05262; NeurIPS 2022); reproduced because CLM-Bench cites but does not restate it. https://arxiv.org/pdf/2202.05262
- **Discrepancy:** Sec. 6.2 cites "Figure 1" for layer-12 metrics whose captions belong to Figure 3 — an internal numbering inconsistency, not a data conflict.

## Verification Log

Independent adversarial fact-check performed 2026-05-31. Sources checked and findings:

**Bibliographic metadata (independently verified online):**
- **arXiv id 2601.17397** — *confirmed online*. WebFetch of https://arxiv.org/abs/2601.17397 and WebSearch both resolve to this exact paper. The HTML mirror https://arxiv.org/html/2601.17397v1 also exists.
- **Title** "CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing" — *confirmed* verbatim from the arXiv abstract page and paper md line 1.
- **Authors** Yucheng Hu, Wei Zhou, Juesi Xiao — *confirmed* from arXiv page and paper md line 3. Affiliation (Tianjin University; School of Future Technology / College of Intelligence and Computing) — *confirmed* from paper md lines 5–7; the arXiv abstract page itself does not print affiliation, so it is sourced from the paper PDF.
- **Submission date 24 Jan 2026, category cs.CL, comment "EACL MME workshop paper"** — *confirmed* via arXiv WebFetch. Workshop tag remains soft (comment-level) until proceedings publish.

**Quantitative claims cross-checked against the source md (papers/CLMBench/CLMBench.md):**
- Headline Llama3-8B/MEMIT/b=1000: zh_score 61.71%, zh-Reliability 63.21%, en-Reliability 19.55%, en_score 68.31%, en-Reliability 75.47%, zh-Reliability 44.42%, trans 30.93%/58.86% — all *match* Tables 1–2 exactly.
- AlphaEdit collapse zh_score 1.84% / en_score 4.04% — *matches* Table 1 (Llama3-8B AlphaEdit, Chinese row).
- Geometric independence (cosine 0.027–0.042, Jaccard <0.32, subspace >60° up to ~80°, RelErr >1.40) and additivity (cosine >0.976, top-5% Jaccard >0.75, RelErr <0.23, subspace 63°→12°) — all *match* Sec. 6.2 verbatim.
- Language-bias correlation −0.092 to 0.042 — *matches* Appendix B.3.
- Layer-12 separation, raw projections (Chinese ≈ −2.0, English ≈ 0.2), projection difference >2.0, layers 9–12, k=10 — *match* Appendix B.1 and Appendix C.
- Domain skew 10.1% (Chinese Literature) vs 0.8% (Modern Chinese History); ~1,100 raw → 1,010 after dedup/review; 2,000 mixed bilingual triplets; eight metrics in four families; EasyEdit — all *match* Secs. 3.1–3.2, 4.1–4.2.
- These metrics are too recent for independent third-party web verification; status is *confirmed against the source md*, not externally corroborated.

**Mathematics:**
- The four geometric metrics (RelErr, Jaccard, CosSim, subspace principal angle via SVD with k=10) reproduced in the analysis *match* the equations in Sec. 6.2 and Appendix C of the source md, including symbols and form.
- ROME closed-form rank-one update is *not stated* in CLM-Bench (it only cites Meng et al., 2022a). The analysis correctly flags this and reproduces the canonical form W = W + Λ(C⁻¹k*)ᵀ, Λ = (v*−Wk*)/((C⁻¹k*)ᵀk*), C = KKᵀ. The arXiv abstract page for 2202.05262 does not expose the equation, but the reproduced form is the established canonical ROME update; symbols and structure are correct.

**Corrections applied during this verification:**
- Sec. "Linear additivity across batch size": the b=100 monolingual Chinese figure was previously written as the muddled "77.96%/74.41% region." 77.96% is actually the b=100 zh-*Locality* value; the zh-*Reliability* at b=100 is 74.41% (Table 2). Corrected to the single correct value 74.41% and added a note that the paper's own Sec. 5.2 prose mis-lists this as 82.83% while its Table 2 says 74.41%.

**No other factual errors found.** arXiv id, title, authors, venue/date, all six headline quantitative blocks, the four delta-metric equations, and the ROME attribution are accurate. Verdict: minor correction (one mislabeled metric value).

**Length note:** the file is ~1.96k words, modestly above the 1,500–1,800 target band (~150–460 words over). Flagged, not padded; trimming the Relevance/Pitfalls section would bring it into band if required.
