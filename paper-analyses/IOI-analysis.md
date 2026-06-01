# Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

**Authors:** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris (Redwood Research), Jacob Steinhardt (Redwood Research / UC Berkeley).
**Venue / year:** ICLR 2023 (poster). **arXiv:** 2211.00593. **Code:** `redwoodresearch/Easy-Transformer` (later TransformerLens).

## Problem and motivation

Mechanistic interpretability tries to explain a model's behavior in terms of its internal components. The authors argue prior work was stuck on two horns: it explained simple behaviors in tiny models, or described complicated behaviors in large models only "with broad strokes." Their goal is a complete, end-to-end reverse-engineering of one natural-language behavior in a real (if small) model, GPT-2 small. The behavior is **indirect object identification (IOI)**: completing "When Mary and John went to the store, John gave a drink to ___" with "Mary." It is linguistically meaningful yet admits a crisp three-step algorithm: (1) list the names that appeared, (2) drop the duplicated name, (3) output the survivor. This makes it a clean "model organism" — narrow enough to fully characterize, rich enough that its phenomena (redundancy, repurposed components, negative components) plausibly generalize.

## Core contribution and precise claims

The headline result is a **circuit of 26 attention heads** — about **1.1% of all (head, token-position) pairs** in GPT-2 small — organized into **7 functional classes** that together perform IOI; the authors claim this is the largest end-to-end reverse-engineering of a natural behavior "in the wild." Methodologically, they introduce **path patching**, a generalization of causal mediation analysis (Vig et al., 2020), supplemented by attention-pattern analysis, embedding-space projections, and activation patching. Evaluatively, they give three falsifiable criteria — **faithfulness, completeness, minimality** — for judging whether a circuit is a real explanation rather than a plausible story.

The seven head classes, traced backward from the logits:

- **Name Mover Heads** (e.g. 9.9, 9.6, 10.0): active at END, attend to the IO token (mean attention 0.59 over p_IOI), copy what they attend to (copy score >95% vs <20% for an average head).
- **Negative Name Mover Heads** (10.7, 11.10): same structure but write in the *opposite* direction (negative copy score 98% vs 12%); hypothesized to "hedge" against high cross-entropy loss.
- **S-Inhibition Heads** (7.3, 7.9, 8.6, 8.10): active at END, attend to S2, write into Name Movers' *queries* to suppress attention to S1 (and S2). Their output carries two disentangled signals — a **token signal** (value of S) and a **position signal** (location of S1).
- **Duplicate Token Heads** (0.1, 3.0): at S2, attend to S1, signal that duplication occurred.
- **Induction Heads** (5.5, 6.9) + **Previous Token Heads** (2.2, 4.11): an alternate duplicate-detection path via the [A][B]…[A] induction mechanism (Elhage et al., 2021; Olsson et al., 2022).
- **Backup Name Mover Heads**: dormant heads that take over name-moving only when the primary Name Movers are ablated.

Three "surprising pitfalls" are emphasized: redundancy/self-repair via Backup heads (so ablation changes the discovered mechanism), known structures (induction heads) repurposed for an unexpected role (here as a positional signal), and heads that write *against* the correct answer.

## Method and mathematics

**Architecture notation (following Elhage et al., 2021).** The residual stream `x_i ∈ R^{N×d}` (N tokens, d the model dimension) is read and written by every layer. Each attention head `h_{i,j}` is parametrized by four matrices and factored into two low-rank `R^{d×d}` operators:

```
W_QK^{i,j} = W_Q^{i,j} (W_K^{i,j})^T       # query-key: sets the attention pattern
W_OV^{i,j} = W_O^{i,j}  W_V^{i,j}           # output-value: sets what is written

A_{i,j} = softmax( x^T W_QK^{i,j} x )        # per-position, causal (unidirectional) softmax
h_{i,j}(x) = M ∘ LN( (A_{i,j} ⊗ W_OV^{i,j}) · x )
```

The layer output is `y_i = Σ_j h_{i,j}(x_i)`; the stream updates as `x_{i+1} = x_i + y_i + y_i'` (MLP term `y_i'`). LayerNorm is `LN(x) = (x − x̄) / sqrt(Σ_i (x_i − x̄_i)^2)` followed by a learned linear map. Logits are `W_U ∘ M ∘ LN(x_L)`.

**Task metrics.** Two scalars quantify performance, averaged over the data distribution p_IOI:
- *Logit difference* = `logit(IO) − logit(S)` (positive ⇒ correct name preferred).
- *IO probability* = `P(IO)` under the model.

Over 100,000 examples, GPT-2 small attains mean logit difference **3.56** (IO over S **99.3%** of the time) and mean IO probability **49%**. The dataset uses **15 templates** (Appendix E) with single-token names drawn from a list of **100 English first names**, with places and objects chosen from a hand-made list of **20 common (single-token) words**; templates come in BABA and ABBA orderings.

**Knockouts and the reference distribution.** A circuit `C` is a subgraph of the computational graph `M`; `C(x)` is defined by *knocking out* every node in `M\C`. Rather than **zero-ablation** (noisy, since 0 is arbitrary and downstream nodes lean on the mean as an implicit bias), they use **mean-ablation**: replace a node's activation by its mean over a reference distribution **p_ABC** — identical generation but with three unrelated random names instead of {IO, S}. This removes IOI-specific information (including the constant fact that a name is duplicated) while preserving grammar; means are taken *per template* so grammatical role stays constant.

**Path patching (the central operator).** Given an original input `x_orig ~ p_IOI`, a counterfactual `x_new ~ p_ABC`, a sender head `h`, and receiver set `R` (keys/queries/values of some heads, or the final residual stream), path patching measures the *direct* effect of `h` on `R` along paths through residual connections and MLPs **but not through other attention heads**. Algorithm 1, five steps:

```
1. Run forward pass on x_new  -> A_new ; on x_orig -> A_orig
2. Freeze every head to its x_orig activation, EXCEPT h, which is set to its x_new activation
3. Forward pass with these frozen/patched values (MLPs & LayerNorm recomputed)
4. Record the recomputed value of each receiver r in R from this pass
5. Final forward pass on x_orig, but overwrite each r in R with its recorded value; read logit diff
```

Because all other heads are frozen, any path `h → p → r` through an intermediate head `p` is removed, isolating the direct `h → r` contribution. Averaged over N > 200 `(x_orig, x_new)` pairs. A large drop in logit difference flags a path critical to the task; the search proceeds backward from Logits → Name Movers' queries → S-Inhibition values/keys → Induction-head keys.

**Validation criteria.** Define `F(C) = E_{X~p_IOI}[ f(C(X); X) ]`, with `f` the IO−S logit difference of circuit `C` on input `X`.
- **Faithfulness:** `|F(M) − F(C)|` small. Here it is **0.46**, i.e. **13%** of `F(M)=3.56`, so `C` recovers **87%** of the model's performance.
- **Completeness:** for every subset `K ⊆ C`, the incompleteness score `|F(C\K) − F(M\K)|` should be small — `C` and `M` must stay similar *under knockouts*, not just at baseline.
- **Minimality:** for every node `v ∈ C` there must exist `K ⊆ C\{v}` with large `|F(C\(K∪{v})) − F(C\K)|`, proving `v` is non-redundant.

**Disentangling the S-Inhibition signal.** Using activation patching across six counterfactual datasets (random name flip; IO↔S1 flip; IO←S2 replacement, and compositions), the logit difference is well approximated by a linear sum `2.31·S_pos + 0.99·S_tok` (mean error 7%), where `S_pos ∈ {+1,−1}` and `S_tok ∈ {+1,0,−1}`. Conclusion: **position signal dominates token signal**, and is relative (insensitive to absolute position when S1–S2 distance is fixed: 3.56 → 3.57).

## Experimental setup and headline numbers

Model: GPT-2 small (L=12 layers, H=12 heads/layer, decoder-only). All interventions act on attention heads; MLPs, LayerNorms, and embeddings are left untouched, except for a note that knocking out **all MLPs after layer 0** breaks the task (logit diff → −1.1), and MLP0 alone flips the sign. Faithfulness 0.46 (87%). Completeness: random-subset and by-class sampling suggested the circuit was complete, but **greedy** optimization found sets with incompleteness up to **3.09 (87%)** — a clear failure. Minimality: every node contributes ≥1% of `F(M)`, but some contributions are small. Adversarial test (double-duplicating IO): the model predicts S over IO **23.4%** of the time (vs **0.7%** on p_IOI), confirming reliance on duplicate detection.

## Stated limitations and threats to validity

The authors are candid. (1) The circuit **fails the hardest completeness test** (greedy `K` gives high incompleteness). (2) Several components stay ununderstood — the *mechanism* of the S-Inhibition attention pattern, the token/position signals, and the role of MLPs/LayerNorms. (3) **Backup/self-repair** means ablation changes the very mechanism under study. (4) Induction heads play a non-canonical role they did not fully characterize. (5) GPT-2 small is orders of magnitude smaller than frontier models; a GPT-2 medium probe already showed more complex direct-effect heads. (6) The adversarial examples could plausibly have been found without the circuit, weakening the downstream-utility argument.

## Relevance to our project (ICCD-3K, Phase 1)

Our study asks where in the network post-training alignment (via any fine-tuning method: SFT, RLHF, DPO, RLVR/GRPO, ...) **selectively** reshapes cultural knowledge and whether that change is recoverable — for each cultural content type, a recoverable late **gate** or an unrecoverable mid-layer **rewrite** — using Indian cultural minimal pairs. Indian culture here is both the controlled probe for that mechanistic question and a genuine subject the study cares about. IOI is the canonical template for the machinery Phase 1 depends on.

**Methods/metrics we borrow.** (1) The **clean-vs-corrupted minimal-pair design** maps directly onto IOI's `x_orig ~ p_IOI` vs `x_new ~ p_ABC`; our "clean prefix vs corrupted prefix + target answer" is the same counterfactual logic, and their template-matched ablation is the right model for fixing grammar while removing only the cultural signal. (2) IOI's **logit-difference** `logit(IO) − logit(S)` is exactly our **per-item log-odds difference** between two target completions, validating our paired per-item scoring. (3) **Path/activation patching** distinguishes "rewrite" from "late gating": patching mid-layer activations and checking whether the cultural log-odds difference recovers tells us *where* the representation lives, while patching query-side late (à la S-Inhibition → Name Mover queries) tests a gating hypothesis. (4) Their **linear-signal decomposition** (`2.31·S_pos + 0.99·S_tok`) models how to separate confounded factors — for us, "cultural content" from "register/format."

**Design constraints it implies.** Use **single-token targets** and **template-matched corruption** so positional alignment holds and means are computed within a grammatical cell — directly informing our **60 cells of 50 items**. Average over hundreds of pairs (they use N>200); our 50/cell with paired t-tests is fine for large effects but borderline for small ones, so pre-register power. **Mean-ablate, not zero-ablate**, over the matched cell.

**Pitfalls it warns against.** **Self-repair/backup**: ablating a mid-layer cultural component may understate its role because downstream heads compensate — so a *null* patching result does not prove "no mid-layer representation"; test completeness, not just faithfulness. **Negative components**: heads writing against the answer mean naive ablation can *increase* a metric, so sign matters. **Faithfulness is necessary but not sufficient** — a high-recovery probe can still be incomplete or non-minimal, so claims about "the" cultural circuit need the completeness/minimality discipline. Finally, mechanisms found in small models may not transfer, so alignment conclusions should be hedged across model sizes.

## Validation notes

- **Title** "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" — *confirmed*. Source: https://arxiv.org/abs/2211.00593
- **Authors** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt — *confirmed* (matches paper and arXiv). Source: https://arxiv.org/abs/2211.00593 and https://openreview.net/forum?id=NpsVSN6o4ul
- **arXiv id 2211.00593** — *confirmed*. Source: https://arxiv.org/abs/2211.00593
- **Venue ICLR 2023 (poster)** — *confirmed*. Source: https://openreview.net/forum?id=NpsVSN6o4ul and https://iclr.cc/virtual/2023/poster/11341
- **26 attention heads, 7 classes, 1.1% of (head, position) pairs** — *confirmed* against paper abstract/Sec. 1 and arXiv abstract. One web summary erroneously said "28 heads"; the paper says 26 (*corrected*). Source: https://arxiv.org/abs/2211.00593
- **Mean logit difference 3.56, IO>S 99.3%, IO prob 49%, over 100,000 examples** — *confirmed* in paper Sec. 2. Source: paper text.
- **Faithfulness 0.46 = 13% of F(M)=3.56, i.e. 87% recovery** — *confirmed* in paper Sec. 4. Source: paper text.
- **Path patching introduced here, generalizing Vig et al. (2020) causal mediation** — *confirmed* in paper Sec. 3 / Appendix B. Source: paper text.
- **Plan citation "Wang et al., ICLR 2023, arXiv:2211.00593"** — *confirmed* correct; no discrepancy.

## Verification Log

Independent adversarial fact-check (2026-05-31). Sources: the arXiv abstract page, the ar5iv full-text HTML, OpenReview, and the local source paper `d:/Mech-Interp-Cultural/papers/IOI/IOI.md`.

**Bibliographic facts — all confirmed.**
- Title verbatim match. Source: https://arxiv.org/abs/2211.00593
- Authors (5, in order: Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt). OpenReview lists the first author as "Kevin Ro Wang." Affiliations Redwood Research (1–4) and UC Berkeley (Steinhardt) match source p.1. Sources: arXiv; https://openreview.net/forum?id=NpsVSN6o4ul
- arXiv id **2211.00593**, submitted **Nov 1, 2022** — confirmed. Source: arXiv.
- Venue **ICLR 2023, poster** — confirmed. Source: OpenReview (ICLR 2023 Conference, poster).

**Quantitative claims — all confirmed against ar5iv full text and source md.**
- 26 attention heads / 7 classes / 1.1% of (head, token-position) pairs (abstract + §1, source line 30).
- Mean logit difference 3.56; IO>S 99.3%; mean IO probability 49%; over 100,000 examples (§2, source line 52).
- Faithfulness |F(M)−F(C)| = 0.46 = 13% of F(M)=3.56; 87% recovery (§4, source line 209).
- Greedy incompleteness up to 3.09 (87% of original logit diff); random/by-class sampling suggested completeness (§4.1, source line 236).
- Name Mover Heads (9.6, 9.9, 10.0): mean attention to IO 0.59; copy score >95% vs <20% (§3.1, source lines 119, 121, 132).
- Negative Name Mover Heads (10.7, 11.10): negative copy score 98% vs 12% (§3.1, source line 134).
- S-Inhibition Heads (7.3, 7.9, 8.6, 8.10); decomposition 2.31·S_pos + 0.99·S_tok, 7% mean error; S_pos∈{+1,−1}, S_tok∈{+1,0,−1}; relative-position invariance 3.56→3.57 (§3.2 / App. A, source lines 157, 367, 384).
- Adversarial double-IO: model predicts S over IO 23.4%; p_IOI baseline 0.7% (Fig. 8 table, source lines 273–275). Note: the §4.4 prose rounds to "23%"; the analysis correctly uses the table value 23.4% and the 0.7% baseline (the 0.4% figure is the additional-S control, not p_IOI).
- All MLPs after layer 0 knocked out → logit difference −1.1; MLP0 alone flips the sign (App. J, source line 605).
- Minimality: every node ≥1% of F(M) (§4.2, source line 253).

**Central mathematics — cross-checked symbol-for-symbol against source.**
- W_QK = W_Q(W_K)^T, W_OV = W_O W_V; A_{i,j}=softmax(x^T W_QK x) (unidirectional); h_{i,j}(x)=M∘LN((A_{i,j}⊗W_OV)·x) — exact match (source line 536).
- LayerNorm Eq. (1) LN(x)=(x−x̄)/sqrt(Σ_i(x_i−x̄_i)^2) — exact match (source line 529).
- Residual update x_{i+1}=x_i+y_i+y_i' with y_i=Σ_j h_{i,j}(x_i); logits W_U∘M∘LN(x_L) — match (Algorithm 2, source lines 561, 563).
- Path patching Algorithm 1 five-step summary, N>200 averaging — match (App. B, source lines 397–408).
- Validation metrics F(C)=E[f(C(X);X)], faithfulness |F(M)−F(C)|, incompleteness |F(C\K)−F(M\K)|, minimality |F(C\(K∪{v}))−F(C\K)| — match (§4–4.2, source lines 207, 226, 242).

**Corrections made this pass.**
- Dataset description (formerly "20 hand-picked single-token places/objects") refined to "places and objects chosen from a hand-made list of 20 common (single-token) words" to match Appendix E (source line 471), which gives 20 words shared across places and objects, not 20 of each.

**No factual errors found** in arXiv id, title, authors, venue/year, head indices, equations, or any load-bearing quantitative claim. The one "28 heads" web-summary error flagged earlier remains a web-source artifact, not present in this analysis (which correctly states 26).

**Word count:** 1909 words (whitespace-delimited), ~6% above the 1500–1800 target band. Content is accurate and non-redundant; flagged here rather than trimmed with filler removal. (The structured summary reported 1795; the live file is slightly longer.)
