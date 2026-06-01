# Technical Analysis: "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods"

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

**Authors:** Fred Zhang (UC Berkeley; work done while interning at Google) and Neel Nanda (Independent).
**Venue / Year:** ICLR 2024 (poster). **arXiv:** 2309.16042 (submitted Sep 27, 2023; v2 Jan 17, 2024). **OpenReview:** Hf17y6u9BC.

## Problem and motivation

Mechanistic interpretability (MI) tries to reverse-engineer a model's internal computation into human-readable algorithms. A prerequisite step is *localization*: pinning down which components (MLP layers, attention heads) are causally responsible for a behavior. The dominant localization tool is **activation patching** (also called causal tracing, interchange intervention, causal mediation analysis, or representation denoising), introduced for LMs by Vig et al. (2020) and Meng et al. (2022). The paper's core observation is sociological as much as technical: nearly every prior paper invents its own variant of patching, with no agreement on (a) how to build a corrupted prompt, (b) what metric measures the patching effect, or (c) whether to patch one layer or a sliding window of several. This lack of standardization means published localization claims could be artifacts of arbitrary hyperparameter choices. The paper is, to the authors' knowledge, the first systematic study of these degrees of freedom, and it issues concrete best-practice recommendations.

## Core contribution and precise claims

The paper identifies three degrees of freedom and studies each empirically across five tasks. The headline claims are:

1. **Corruption method matters.** Gaussian Noising (GN) and Symmetric Token Replacement (STR) can yield *inconsistent* localization and circuit-discovery results. GN tends to push the model out-of-distribution (OOD), which the authors argue breaks internal mechanisms and can produce unreliable or even illusory localization. **Recommendation: prefer STR.**
2. **Metric matters.** Probability and logit difference disagree; crucially, probability is non-negative and therefore *cannot* detect components that *hurt* the correct answer when corruption already drives the answer probability near zero. **Recommendation: prefer logit difference; KL divergence is also reasonable.**
3. **Sliding-window patching inflates localization** relative to summing single-layer effects, because joint patching captures non-linear cross-layer interactions. **Recommendation: try single-layer first; interpret window results as joint effects.**
4. A fourth, often-overlooked knob: **which token(s) you corrupt** changes what information patching traces, and thus the discovered circuit.

## Method and mathematics

### The patching procedure

Patching uses a clean prompt `X_clean` ("The Eiffel Tower is in") with answer `r` ("Paris"), a corrupted prompt `X_corrupt` ("The Colosseum is in") with answer `r'` ("Rome"), and three forward passes:

```
(1) Clean run:     run model on X_clean, cache activations of target components.
(2) Corrupted run: run model on X_corrupt, record outputs.
(3) Patched run:   run model on X_corrupt, but overwrite ONE component's
                   activation with its cached value from the clean run.
```

Subscripts `cl`, `*`, `pt` denote the clean, corrupted (the `*`), and patched runs. The intuition: corruption hurts performance, patching restores it; the **patching effect** measures how much a component's clean activation restores the corrupted run, indicating that component's importance. Iterating over all components yields the localization heatmap (Figure 1b).

### Corruption operators

- **Gaussian Noising (GN):** add noise `ε ~ N(0, ν)` to the token embeddings of the key tokens (e.g., the subject), where `ν = 3 × (standard deviation of token embeddings over the dataset)`. GN does not define a clean alternative answer `r'`.
- **Symmetric Token Replacement (STR):** swap the key tokens for semantically related tokens of *equal sequence length* ("The Eiffel Tower" → "The Colosseum"). STR yields an in-distribution `X_corrupt` that is identically distributed to a fresh clean draw, and it supplies a well-defined `r'`.

### Metrics (the central definitions)

```
Probability:      P(r)                         e.g. P("Paris")
  patching effect = P_pt(r) − P_*(r)

Logit difference: LD(r, r') = Logit(r) − Logit(r')   e.g. Logit("Paris") − Logit("Rome")
  patching effect = LD_pt(r, r') − LD_*(r, r')
  normalized by  (LD_cl(r, r') − LD_*(r, r'))  so it lies in [0, 1]:
     1 = fully restored (clean) performance,  0 = corrupted performance.

KL divergence:    D_KL(P_cl || P)              (clean output distribution as reference)
  patching effect = D_KL(P_cl || P_*) − D_KL(P_cl || P_pt)
```

Symbols: `Logit(·)` is the pre-softmax logit at the final position; `P_cl, P_*, P_pt` are the full output distributions; `r` correct token, `r'` counterfactual token. Because GN lacks a native `r'`, the paper reuses STR's `r'` for GN's logit difference to keep comparisons fair. Layers are zero-indexed `0..L−1`.

### The non-negativity argument (why probability misses negative heads)

This is the paper's cleanest piece of analysis. In the IOI STR setting the corrupted-run probability of the original IO averages 0.03, so the probability patching effect `P_pt(IO) − P_*(IO) ≥ −0.03` (since `P_pt ≥ 0`). The detection threshold is 2 SD below the mean (mean 0.003, SD 0.015, i.e., −0.027). A Negative Name Mover such as head 11.10 (effect −0.022) therefore *cannot* clear the bar — to be detected its `P_pt(IO)` must be near 0, which is hard. Under the fully-random `p_ABC` corruption (S1, S2, IO all replaced), `P_*(IO) ≈ 5e-4`, and probability detects *neither* negative head, while logit difference still does. The general lemma: **probability cannot surface negative components when corruption drives the correct-token probability to near zero**, because the effect is floored at `−P_*(r)`.

### Sliding window vs. summation

For a window of size `w`, sliding-window patching restores `w` adjacent MLP layers jointly. The comparison baseline patches each layer alone and *sums* the single-layer effects over the window (e.g., the effect for layer 4 = sum of single-layer effects of layers 2–6 for `w = 5`). The window method gave 1.40×, 1.75×, 1.59× the peak of summation at window sizes 3, 5, 10, and "typically at least 20% more peak effect." The authors attribute the gap to two non-linear effects: (i) the window suppresses corrupted-information flow that no single layer can block, and (ii) several layers jointly perform a computation no single layer achieves alone.

### Transformer notation (Appendix A)

Following Elhage et al. (2021): input `x_0 ∈ R^{N×d}` (sum of token + position embeddings) is the residual stream. Per-layer attention output decomposes as `y_i = Σ_{j=1}^H h_{i,j}(x_i)`, each head with `W_Q^{i,j}, W_K^{i,j}, W_V^{i,j} ∈ R^{d × d/H}` and `W_O`. The attention pattern is `A^{i,j} = softmax((xW_Q)(xW_K)^T / sqrt(d/H) + M)`, `M` the causal mask.

## Experimental setup and headline numbers

**Tasks:** factual recall (Meng et al. 2022), IOI circuit (Wang et al. 2023), greater-than (Hanna et al. 2023), Python docstring (Heimersheim & Janiak 2023), arithmetic (Stolfo et al. 2023). **Models:** GPT-2 small/large/XL, GPT-J 6B — all decoder-only, ≤ 6B params. Implemented in TransformerLens and Wang et al.'s Easy-Transformer codebase.

- **PairedFacts dataset:** 145 length-matched prompt pairs from CounterFact and Known1000. GPT-2 XL: 49.0% correct-token probability, 6.85 logit difference; GPT-2 large: 41.1% / 5.88; GPT-J: 50.1% / 7.36.
- **Factual recall, MLP patching:** under GN there is a clear peak near layer 16; under STR that peak is "not salient at all." Across window sizes the GN peak is **2×–5×** higher than STR.
- **Metric/position ratio (last subject token vs middle subject tokens):** STR — 4.33× (probability) > 1.22× (logit diff); GN — 1.74× (probability) > 0.77× (logit diff). Probability systematically over-weights the last subject token.
- **IOI:** averaged over 500 prompts. Clean `P(IO) = 0.481`; GN corrupted `P_*(IO) = 0.129`. STR and GN detect *different* head sets (Table 1; head classes NM, DT, S-Inhibition, Negative NM, Induction). On clean prompts Name Movers put 0.58 attention probability on IO; under GN this splits across IO and S1 (0.26 / 0.21), evidence of OOD disruption. Restoring S-Inhibition head values recovers IO logit under STR (logit diff 1.04) but not GN (0.49). The 3 Name Mover heads (9.6, 9.9, 10.0) are mostly *missed* when corrupting S2 but recovered when corrupting S1+IO — confirming "which token to corrupt" matters.
- **Docstring:** the 4-layer model solves the task at 56% accuracy, logit diff 0.5; heads 3.0 and 3.6 are consistently the top detections.
- **Arithmetic (GPT-J):** STR gives sharper MLP concentration than GN (up to 4×) for add/subtract — the *opposite* direction from factual recall — showing the GN-vs-STR gap is unpredictable. Stolfo et al.'s ratio metric is inflated by a tiny `P_*(r)` denominator under STR.

## Limitations and threats to validity

The authors are explicit: experiments are confined to decoder-only models ≤ 6B parameters; larger models and other architectures are untested. They only test denoising (corrupt→clean overwrite), not the reverse noising direction. The OOD-breaks-mechanisms claim for GN is supported by "tentative evidence" (the NM attention shift), not proof. Their own circuit recovery is incomplete (Name Movers missed under S2 corruption), implying manual inspection and path patching remain necessary. The PairedFacts set is small (145 pairs).

## Relevance to our Phase 1 (ICCD-3K)

Our study asks where in the network post-training alignment (via any fine-tuning method: SFT, RLHF, DPO, RLVR/GRPO, instruction-tuning) **selectively** reshapes cultural knowledge and whether that change is recoverable — for each cultural content type, a recoverable late **gate** or an unrecoverable mid-layer **rewrite** — using Indian-cultural minimal pairs, where Indian culture is both the controlled probe for this mechanistic question and a genuine subject the study cares about. Phase 1 builds a 3,000-item probe set (clean prefix vs corrupted prefix + target answer; per-item log-odds difference; 60 cells × 50 items; paired t-tests). This paper directly governs how we should design it.

**Methods/metrics we borrow.** Our per-item log-odds difference is essentially the paper's **logit difference metric** at the answer token, and the paper gives us a principled reason to prefer it over raw probability: only logit difference (or a paired clean-vs-corrupt log-odds contrast) can surface components that *suppress* a culturally-correct answer — exactly the "gating" hypothesis. If alignment installs a late-layer head that down-weights a mid-layer cultural feature, a probability metric would floor out and hide it; the non-negativity argument is the formal warning. We should report a normalized effect in [0,1] (clean = 1, corrupt = 0) so cells are comparable.

**Design constraints it implies.** Our "clean prefix vs corrupted prefix" construction is **STR**, the paper's recommended corruption — and our minimal pairs must be *length-matched and in-distribution* (same tokenizer sequence length, semantically parallel cultural swaps), mirroring PairedFacts. This keeps corrupted activations in-distribution and avoids the OOD confound. We should run both directions of each pair (each item serving as clean and corrupt), as the paper does, to symmetrize. Use a fixed counterfactual target `r'` per pair so log-odds is well defined. With 50 items/cell, the paper's 145-pair and 500-prompt scales suggest our per-cell n is adequate for paired t-tests but we should expect noisy single-cell estimates.

**Pitfalls it warns against.** (1) Do **not** use Gaussian noising on embeddings to corrupt cultural prefixes — it would break the very mid-layer mechanisms we want to measure and could manufacture an illusory "rewrite" signal. (2) **Which token we corrupt determines what we trace**: corrupting the cultural-entity token vs. context tokens will localize different things, so our minimal-pair swap point must be chosen deliberately and held fixed across cells. (3) Avoid **sliding-window MLP patching** as a primary readout for the rewrite-vs-gate question — joint windows inflate and smear mid-layer peaks, which would bias us toward a spurious "mid-layer rewrite" conclusion. Patch single layers first; only widen windows when single-layer effects are weak, and interpret them as joint effects. (4) Probability/last-token position bias (the 4.33× vs 1.22× result) means metric choice alone can shift where we believe processing happens — a confound we must control by fixing logit difference across all 60 cells.

## Verification Log

Independent adversarial fact-check performed 2026-05-31. Every load-bearing claim was cross-checked against the source paper markdown at `d:/Mech-Interp-Cultural/papers/ActivationPatching/ActivationPatching.md` (full text, all 859 lines / appendices A–I) and against live web sources. Result: no factual errors found; all numbers, equations, authors, ids, and dates match the source. No corrections to the body were required.

**Bibliographic facts (web-verified):**
- **Title** "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods" — **confirmed.** Sources: https://arxiv.org/abs/2309.16042 ; paper md line 1.
- **Authors** Fred Zhang (UC Berkeley; "work done while interning at Google") and Neel Nanda (Independent) — **confirmed.** Sources: https://arxiv.org/abs/2309.16042 (WebFetch) ; paper md lines 3–9.
- **arXiv id** 2309.16042 — **confirmed.** Sources: https://arxiv.org/abs/2309.16042 ; WebSearch (Google Scholar, Hugging Face, Semantic Scholar, ADS all return this id).
- **Venue / dates** ICLR 2024 poster; v1 submitted 27 Sep 2023, v2 17 Jan 2024 — **confirmed.** Source: arXiv abstract page (WebFetch reported "Submission Date: September 27, 2023 (v1); Revised January 17, 2024 (v2); Venue: ICLR 2024"). Also https://openreview.net/forum?id=Hf17y6u9BC ; https://iclr.cc/virtual/2024/poster/18984 .

**Quantitative / equation facts (verified line-by-line against paper md):**
- PairedFacts = 145 length-matched pairs from CounterFact + Known1000; GPT-2 XL 49.0% / 6.85 LD, GPT-2 large 41.1% / 5.88, GPT-J 50.1% / 7.36 — md lines 420–424. **Confirmed exactly.**
- Factual-recall GN peak (≈layer 16) 2×–5× higher than STR across window sizes — md lines 122–124. **Confirmed.**
- Position ratio (last subject token vs middle): STR 4.33× (prob) > 1.22× (LD); GN 1.74× (prob) > 0.77× (LD) — md lines 191–192. **Confirmed exactly.**
- IOI: 500 prompts; clean P(IO)=0.481; GN corrupted P_*(IO)=0.129 — md lines 436, 118. **Confirmed.**
- NM attention to IO 0.58 (clean) → 0.26/0.21 IO/S1 split (GN) — md line 169. **Confirmed.** S-Inhibition value restoration: STR LD 1.04, GN 0.49 — md lines 171–175. **Confirmed.**
- Non-negativity argument: STR corrupted P(IO)≈0.03; threshold −0.027 = mean 0.003 − 2×SD 0.015; NNM 11.10 effect −0.022; p_ABC P_*(IO)≈5e-4 — md lines 222–231. **Confirmed exactly** (−0.027 = 0.003 − 2·0.015 checks).
- Name Mover heads = 9.6, 9.9, 10.0; Negative NM = 10.7, 11.10; S2 corruption misses ≥2 of 3 NMs, S1+IO recovers them — md lines 584, 619, 288. **Confirmed.** (Body text correctly says "mostly missed"; the paper's exact figure is "at least 2 out of the 3," with 9.6 and 10.0 missed by all metric/method combos.)
- Sliding window vs summation: 1.40× / 1.75× / 1.59× peak at window 3 / 5 / 10; "typically at least 20% more peak" — md lines 217, 245. **Confirmed.**
- Docstring: 4-layer attn-only model, 56% accuracy, LD 0.5; heads 3.0 and 3.6 consistently top — md lines 520, 541. **Confirmed.**
- Arithmetic (GPT-J): STR sharper than GN up to 4× for add/subtract, opposite direction to factual recall; Stolfo metric (eq. 2) first term inflated by tiny P_*(r) denominator under STR — md lines 463, 470, 472. **Confirmed** (Stolfo equation transcribed correctly).
- GN noise ν = 3× std of token embeddings; STR equal-length swap with well-defined r'; same r' reused for GN logit diff; layers zero-indexed 0..L−1 — md lines 412–416. **Confirmed.**
- Transformer notation (Elhage et al. 2021): y_i = Σ_j h_{i,j}(x_i); A = softmax((xW_Q)(xW_K)^T / √(d/H) + M); residual update x + Concat[A·V]W_O — md lines 402–408. **Confirmed exactly** (symbols and form correct).

**Notes / caveats:**
- **Task-brief examples** (ROME rank-one update, difference-in-means refusal direction, MDL probing objective) are **not present in this paper** and were correctly flagged in the structured summary as generic methodology examples, not as content of this work — **no fabrication detected.**
- The Semantic Scholar record (c16c05ca...) was not independently loadable (empty fetch), but venue/year/authors/id are triangulated by arXiv, OpenReview, ICLR proceedings, and four indexing sites, so this does not affect any verdict.
- **Word count:** the file is ≈1900–2000 words (raw `wc -w` 2019; ≈1901 excluding fenced code blocks), which is **above the 1500–1800 target band.** Flagged per instruction; not padded or cut, since the content is dense and accurate and trimming would remove substance. If strict compliance is required, the "Method and mathematics" subsections could be tightened.
