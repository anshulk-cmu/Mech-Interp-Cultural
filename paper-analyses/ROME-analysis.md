# Locating and Editing Factual Associations in GPT (ROME): Technical Analysis

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

**Paper:** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." NeurIPS 2022 (Advances in Neural Information Processing Systems, vol. 35). arXiv:2202.05262 (v1 Feb 10 2022; v5 Jan 13 2023). Project page: https://rome.baulab.info/.

## Problem and motivation

The paper asks a concrete mechanistic question: *where* inside an autoregressive transformer does a factual association live, and is that location specific enough that the fact can be edited by changing a small set of weights? Prior knowledge-probing work on masked models (Petroni et al. 2019; Geva et al. 2021; Dai et al. 2022) had implicated MLP layers but worked behaviorally or on BERT; GPT-style models, with unidirectional attention, were under-explored. The authors want a *causal* localization (not a correlational probe) and then a *constructive* test: if the storage site is correctly located, a new fact can be written into it surgically, achieving both **generalization** (the edit survives paraphrase) and **specificity** (unrelated facts untouched). Causal Tracing and ROME are designed so that each validates the other.

## Core contributions and precise claims

1. **Causal Tracing**, a paired-intervention method using causal mediation analysis (Pearl 2001; Vig et al. 2020b) that measures the indirect effect of individual hidden states on a factual prediction.
2. The empirical finding of a **decisive "early site": mid-layer MLP modules at the last subject token** mediate factual recall, distinct from the unsurprising late site at the final token.
3. **ROME (Rank-One Model Editing)**, a closed-form rank-one update to one MLP down-projection matrix that inserts a new (subject, relation, object) association.
4. **CounterFact**, a 21,919-record dataset of difficult counterfactual assertions with metrics that separate superficial regurgitation from genuine, generalizing edits.
5. The claim that ROME *simultaneously* achieves generalization and specificity where fine-tuning, hypernetwork (KE, MEND), and neuron-attribution (KN) baselines sacrifice one or the other.

## Method and mathematics

### Transformer notation

A model `G: X → Y` over vocabulary `V` maps tokens `x = [x_1,...,x_T]` to a distribution `y ∈ R^|V|`. Hidden states evolve as a residual stream (`H` = hidden dim, `L` layers):

```
h_i^(l) = h_i^(l-1) + a_i^(l) + m_i^(l)
a_i^(l) = attn^(l)( h_1^(l-1), ..., h_i^(l-1) )
m_i^(l) = W_proj^(l) · σ( W_fc^(l) · γ( a_i^(l) + h_i^(l-1) ) )
```

Here `a_i^(l)` is the attention contribution, `m_i^(l)` the MLP contribution, `σ` a rectifying nonlinearity, `γ` layernorm, and `W_fc`, `W_proj` the two MLP matrices. `h_i^(0) = emb(x_i) + pos(i)`; output `y = decode(h_T^(L))`. A fact is a tuple `t = (s, r, o)` (subject, relation, object), elicited by a natural-language prompt `p` describing `(s, r)`.

### Causal Tracing

Three runs over a known factual prompt:

- **Clean run:** collect all `{h_i^(l)}`; record `P[o]`.
- **Corrupted run:** immediately after embedding, set `h_i^(0) := h_i^(0) + ε` for every token `i` in the subject span, with `ε ~ N(0; ν)`; record `P_*[o]`. In practice `ν = 3·σ_t`, three times the observed std of token embeddings, repeated 10× per prompt.
- **Corrupted-with-restoration run:** run corrupted, but at one `(î, l̂)` force the hidden state back to its clean value `h_i^(l)`; let computation continue. Record `P_{*, clean h_i^(l)}[o]`.

The metrics:

```
Total Effect    TE = P[o] − P_*[o]
Indirect Effect IE = P_{*, clean h_i^(l)}[o] − P_*[o]
```

Averaged over a sample of statements these give ATE and **AIE** (Average Indirect Effect), the central tracing quantity. A modified graph that *severs* MLP (freezing `m_i^(l)` in the corrupted state) isolates MLP-specific path effects (Pearl 2001).

### The associative-memory view and the rank-one update

ROME treats `W_proj^(l)` as a **linear associative memory** (Kohonen 1972; Anderson 1972): a linear map storing key matrix `K` to value matrix `V` via `WK ≈ V`, minimized by the pseudoinverse `W = VK^+`. To insert one new pair `(k_*, v_*)` while preserving the rest, solve a least-squares problem with an equality constraint:

```
minimize ‖ Ŵ K − V ‖   subject to   Ŵ k_* = v_*
solution:  Ŵ = W + Λ (C^{-1} k_*)^T
where      Λ = (v_* − W k_*) / ( (C^{-1} k_*)^T k_* )
           C = K K^T  (uncentered covariance of keys)
```

`C` is **pre-cached** from ~100,000 hidden-state key vectors sampled from all tokens of Wikipedia text (2020-05-01 snapshot, float32). The derivation (Appendix A) sets up the Lagrangian `L(Ŵ, Λ) = ½‖ŴK−V‖_F^2 − Λ^T(Ŵk_* − v_*)`, differentiates, subtracts the normal equations `WKK^T = VK^T`, and yields the rank-one form `(Ŵ−W)KK^T = Λk_*^T`. The update is genuinely rank one: `Λ` is a vector and `(C^{-1}k_*)^T` a row vector.

**Step 1 — key `k_*` (selects the subject).** Read the post-nonlinearity MLP input at layer `l*`, last subject token `i`, averaged over `N` context prefixes:

```
k_* = (1/N) Σ_j k(x_j + s),   k(x) = σ( W_fc^(l*) · γ( a_[x],i^(l*) + h_[x],i^(l*-1) ) )
```

In practice 20 prefixes (ten length-5, ten length-10) make `k_*` robust to context. (The main text says 50 sequences of length 2–10; Appendix E.5 specifies the 20-prefix configuration actually used, and reports a no-prefix ablation S′=86.1 vs S=89.2.)

**Step 2 — value `v_*` (recalls the new object).** Optimize a vector `z` (not weights):

```
v_* = argmin_z  (1/N) Σ_j [ −log P_{G(m_i^(l*) := z)}[ o* | x_j + p ] ]          (a) maximize o*
                       + D_KL( P_{G(m_i'^(l*) := z)}[x | p'] ‖ P_G[x | p'] )    (b) essence drift
```

Term (a) makes the network predict the target object `o*` when `z` is substituted as the MLP output at the subject's last token. Term (b) is a KL penalty against the unmodified model on a prompt `p'` of the form "{subject} is a", preserving the subject's "essence." Adam, lr 0.5, weight decay 1.5e-3, KL scale `λ = 100`, ≤20 steps, early stop at loss 5e-2.

**Step 3 — insert.** Plug `(k_*, v_*)` into the rank-one update at layer **18** (GPT-2 XL), the MLP center of causal effect. One edit ≈ 2 s on an A6000.

## Experimental setup and headline results

**Models:** GPT-2 XL (1.5B), GPT-J (6B); tracing also on GPT-NeoX (20B), GPT-2 Medium (345M) and Large (774M).

**Causal tracing (1000 known facts, GPT-2 XL):** average correct-object probability 27.0%, dropped to 8.47% under corruption. **ATE = 18.6%**; peak individual-state **AIE = 8.7% at layer 15**, last subject token. Decomposed: **MLP AIE peaks at 6.6%**, attention at the last subject token only **1.6%** (attention dominates at the last token before prediction). Causal Tracing beats integrated gradients for revealing this global structure.

**zsRE editing (10,000 records, GPT-2 XL, Table 1):** ROME Efficacy **99.8**, Paraphrase **88.1**, Specificity **24.2** — competitive with custom-trained MEND-zsRE (99.4 / 99.3 / 24.1) and FT (99.6 / 82.1 / 23.2). zsRE specificity is shown to be insensitive to model damage.

**CounterFact (7,500-record GPT-2 XL, 2,000-record GPT-J test sets, Table 4).** Metrics: Efficacy Score/Magnitude (ES/EM, `P[o*]>P[o^c]`), Paraphrase (PS/PM), Neighborhood (NS/NM, specificity on nearby true subjects), generation Consistency (RS, TF-IDF cosine to reference texts), Fluency (GE, bi/tri-gram entropy `−Σ_k f(k) log_2 f(k)`), and an overall **harmonic mean S** of ES/PS/NS. ROME on GPT-2 XL: **S = 89.2** (ES 100.0, PS 96.4, NS 75.4) vs FT 65.1, FT+L 66.9, KE 52.2, MEND 57.9, KN 35.6, unedited 30.5. On GPT-J: **S = 91.5** vs FT 25.5, FT+L 68.7, MEND 63.2. ROME generalization peaks at **layer 18**, matching the traced early site. Human evaluation (15 volunteers, 50 facts, 150 judgments): raters were **1.8× more likely** to call ROME more consistent than FT+L, but **1.3× less likely** to call it more fluent — ROME costs some fluency the entropy metric misses.

**CounterFact composition (Table 2):** 21,919 records, 20,391 subjects, 749 objects, 42,876 paraphrase prompts, 82,650 neighborhood prompts, 62,346 generation prompts.

## Stated limitations and threats to validity

ROME edits **one fact at a time** and is a *probe of mechanism, not a deployment tool*; scaling is left to follow-up (MEMIT, Meng et al. arXiv:2210.07229). Edits are **directional**: "Space Needle is in Seattle" and its inverse are stored separately, needing two edits. The work covers only factual associations, not logical/spatial/numerical knowledge, and the vector-space structure of attributes is not fully understood. An edited model will **confabulate** plausible-but-false downstream facts, limiting use as a knowledge source. Methodologically, tracing depends on the corruption rule and noise scale `ν`: too-small noise yields negligible total effects; the localized-storage hypothesis also rests on the layer-order-exchange assumption (Zhao et al. 2021), which is an assumption, not a proof.

## Relevance to our project (ICCD-3K, Phase 1)

ICCD-3K's design is explicitly modeled on CounterFact, and ROME supplies both the template and the cautions. **Methods/metrics we borrow.** The minimal-pair structure (clean prefix vs corrupted twin differing in one anchor, with a target answer) is the analogue of CounterFact's requested-rewrite + neighborhood construction. Our per-item log-odds difference `ΔL` is the sequence-probability cousin of ROME's `P[o*] − P[o^c]` magnitude metrics (EM/PM/NM) and of the indirect-effect probability differences in tracing; reading the target log-probability immediately before the answer token mirrors ROME's measurement point. CounterFact's split into Efficacy / Generalization / Specificity (neighborhood) / Consistency / Fluency, aggregated by a tradeoff-aware harmonic mean `S`, is the precedent for our multi-axis reporting: Axis A regional binding parallels neighborhood specificity, Axis B flattening parallels consistency/essence, Axis C parallels late-policy behavior. The **last-subject-token / layer-18 finding** motivates Phase 2 probing and Phase 4 patching: ROME tells us to read the residual stream at the subject's last token and that mid layers (~15–18 in GPT-2 XL) are where MLPs write factual content, so probing should not expect cultural facts only at the final-token late site.

**Design constraints it implies.** (1) *Pre-cache covariance from background text, not the probe set* — ROME computes `C` from 100k Wikipedia keys; our Crosscoders (Phase 3) must likewise train on a large background corpus, never on the 3,000 items. (2) *Token-position alignment matters* — ROME reads `k_*` at a precise index, validating our bare-prefix, matched-suffix decision (Filter F5) so positions correspond across checkpoints. (3) *Causal mediation, not gradient saliency* — ROME shows integrated gradients miss the global structure, supporting our reliance on activation patching and DLA.

**Pitfalls it warns against.** (a) **Specificity metrics can be insensitive** — ROME found zsRE specificity blind to bleedover because unrelated facts are too far in fact-space; the fix was *nearby* neighborhood subjects, so our Axis A corruption must use same-category, close-in-latent-space distractors (the plan's cross-anchor swap is exactly this lesson). (b) **Regurgitation vs genuine change** — CounterFact exists because easy benchmarks reward surface word changes (Hase et al. 2021); our base-model validation threshold (`ΔL > 1.0` nat) plays the analogous filtering role. (c) **Essence drift / fluency loss** — ROME's KL term and human-eval fluency penalty warn that interventions degrade coherence in ways automatic metrics miss. (d) **Directionality** — facts are stored asymmetrically, so our prefix framing fixes one direction. (e) ROME is single-fact and confabulates, but it demonstrates mid-layer MLP weights *can* store and be rewritten — exactly the rewriting signature Phase 4 must distinguish from late gating.

## Validation notes

- **Title** "Locating and Editing Factual Associations in GPT" — **confirmed** (arXiv:2202.05262 abstract page; paper text line 1).
- **Authors** Kevin Meng (MIT CSAIL), David Bau (Northeastern), Alex Andonian (MIT CSAIL), Yonatan Belinkov (Technion) — **confirmed** (arXiv:2202.05262; NeurIPS proceedings; paper text). Equal contribution Meng/Bau.
- **arXiv id 2202.05262** — **confirmed** (https://arxiv.org/abs/2202.05262); v1 2022-02-10, v5 2023-01-13; categories cs.CL, cs.LG.
- **Venue NeurIPS 2022** (Advances in NeurIPS vol. 35) — **confirmed** (https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html; GitHub kmeng01/rome).
- **Models GPT-2 XL (1.5B), GPT-J (6B)** — **confirmed** (GitHub kmeng01/rome README; paper text).
- **CounterFact = 21,919 records** — **confirmed** from paper Table 2 (the GitHub README did not state the count; cross-checked internally).
- **AIE 8.7% @ layer 15, ATE 18.6%, MLP 6.6% / attn 1.6%, edit layer 18** — **confirmed** within paper text (Sections 2.2, 3.4, App. E.5); these are paper-internal numbers not independently web-verifiable.
- **zsRE ROME 99.8/88.1/24.2; CounterFact S 89.2 (GPT-2 XL) / 91.5 (GPT-J)** — **confirmed** (paper Tables 1, 4).
- **Follow-up MEMIT arXiv:2210.07229** (Meng, Sen Sharma, Andonian, Belinkov, Bau 2022) — **confirmed** (paper references; cited as scalable multi-edit method).

## Discrepancy with the project plan

- The plan (Phase-1 spec, Section 2.5) states ROME "studied roughly 1,000 systematically chosen CounterFact items rather than all 21,000." This conflates two distinct samples. The **1,000** figure is the Causal-Tracing statement sample (Section 2.2 / Appendix B.1), drawn for the *tracing* experiments, not from CounterFact. The **CounterFact editing** evaluation used **7,500** records (GPT-2 XL) and **2,000** records (GPT-J) test sets out of the full **21,919**. The plan's "~21,000" for the full set is approximately right (21,919), but "roughly 1,000 CounterFact items" is inaccurate as a description of the edit evaluation. The precedent argument still holds (ROME used a focused subset, not the whole set), but the specific item count should be corrected to "7,500 / 2,000 of 21,919" for the editing study and "1,000 facts" for causal tracing.
- Plan citation "Meng et al. 2022, arXiv:2202.05262, NeurIPS 2022, CounterFact" — **all confirmed**, no correction needed.

## Verification Log

Independent adversarial fact-check performed 2026-05-31. Sources: arXiv abstract page (https://arxiv.org/abs/2202.05262), arXiv PDF (https://arxiv.org/pdf/2202.05262), NeurIPS 2022 proceedings page, and a line-by-line cross-check against the source paper markdown at `d:/Mech-Interp-Cultural/papers/ROME/ROME.md`.

**Bibliographic metadata — all CONFIRMED, no corrections.**
- Title "Locating and Editing Factual Associations in GPT" — confirmed (arXiv abstract; source line 1).
- Authors Kevin Meng (MIT CSAIL), David Bau (Northeastern), Alex Andonian (MIT CSAIL), Yonatan Belinkov (Technion–IIT); equal contribution Meng/Bau — confirmed (arXiv; source lines 3-11).
- arXiv id 2202.05262; v1 2022-02-10, v5 2023-01-13; categories cs.CL, cs.LG — confirmed (arXiv abstract page).
- Venue NeurIPS 2022 (Advances in NeurIPS vol. 35) — confirmed (arXiv WebFetch + NeurIPS proceedings).

**Quantitative claims — all CONFIRMED against the source paper, no corrections.**
- Causal tracing: clean correct-object prob 27.0% → 8.47% corrupted (source App. B.1, lines 437/439); ν = 3σ_t, repeated 10× (line 439); ATE 18.6%, AIE 8.7% at layer 15, MLP AIE 6.6%, attention 1.6% (lines 84/86); 1000-fact tracing sample (lines 84/437). All correct.
- Equations: residual stream Eqn 1 (source line 53), rank-one update Ŵ = W + Λ(C⁻¹k_*)ᵀ with Λ = (v_*−Wk_*)/((C⁻¹k_*)ᵀk_*) and C = KKᵀ (Eqn 2, lines 120/123/427), k_* averaging Eqn 3 (line 132), v_* objective with KL essence-drift term Eqn 4 (line 140) — every symbol and form matches the source.
- ROME implementation: edit at layer 18 (App. E.5, line 740); C pre-cached from 100,000 Wikipedia keys, 2020-05-01 snapshot, all tokens, float32 (line 742); 20 prefixes (ten len-5, ten len-10) with main-text "50 sequences length 2–10" discrepancy correctly noted (lines 135/744); no-prefix ablation S′=86.1 vs S=89.2 (line 746); Adam lr 0.5, wd 1.5e-3, KL λ=100, ≤20 steps, early stop 5e-2, ~2 s on A6000 (lines 750/752). All correct.
- zsRE Table 1: ROME 99.8/88.1/24.2, MEND-zsRE 99.4/99.3/24.1, FT 99.6/82.1/23.2 (lines 162/167/168). Correct.
- CounterFact Table 4: ROME GPT-2 XL S=89.2 (ES 100.0, PS 96.4, NS 75.4); GPT-J S=91.5; FT 65.1, FT+L 66.9, KE 52.2, MEND 57.9, KN 35.6, unedited 30.5; GPT-J FT 25.5, FT+L 68.7, MEND 63.2 (lines 231-244). All correct.
- CounterFact composition Table 2: 21,919 records / 20,391 subjects / 749 objects / 42,876 paraphrase / 82,650 neighborhood / 62,346 generation prompts (lines 202-208). All correct.
- Human eval: 15 volunteers, 50 facts, 150 evaluations; ROME 1.8× more likely judged more consistent, 1.3× less likely more fluent (source lines 256/279/958). Correct (the "150 judgments" figure verified in App. J, Figure 26 caption, line 958).
- Follow-up MEMIT (Meng, Sen Sharma, Andonian, Belinkov, Bau 2022) = arXiv:2210.07229 — citation existence confirmed in source limitations (lines 283/301); the specific 2210.07229 id is from the analyst's external lookup and is the correct MEMIT id.

**Discrepancy-with-plan analysis (1,000 tracing facts vs 7,500/2,000 CounterFact edit sets out of 21,919)** — independently re-verified: the 1,000 figure is the causal-tracing sample (App. B.1, line 437), and the edit evaluation used 7,500 (GPT-2 XL) / 2,000 (GPT-J) records (line 248). The analyst's correction is accurate.

**One non-error noted, not changed:** the source is internally inconsistent on GPT-2 Medium size — Figure 9 caption (line 502) says "334 million" while App. F (line 781) says "345M." The analysis uses 345M, matching the paper's body text and the conventional value; left as-is.

**Corrections applied to this file:** none. Every checked fact, number, and equation was already accurate.

**Length flag:** This file is approximately 2,134 words, which is materially ABOVE the requested 1,500–1,800-word band (~330 words / ~19% over). Content is accurate and non-redundant, so it was not trimmed here, but the overage is flagged for the author's attention. (Note: the upstream structured summary recorded word_count 1986, which is also inaccurate; the measured count is ~2,134.)
