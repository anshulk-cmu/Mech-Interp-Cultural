# Refusal in Language Models Is Mediated by a Single Direction — Technical Analysis

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

**Authors:** Andy Arditi (Independent), Oscar Obeso (ETH Zürich), Aaquib Syed (U. Maryland), Daniel Paleka (ETH Zürich), Nina Panickssery (Anthropic), Wes Gurnee (MIT), Neel Nanda. **Venue:** NeurIPS 2024 (poster), *Advances in Neural Information Processing Systems 37*. **arXiv:** 2406.11717 (submitted 17 Jun 2024). All four facts verified online; see Validation notes.

## Problem and motivation

Chat LLMs are fine-tuned to be both helpful and harmless: they obey benign requests and refuse harmful ones (Bai et al. 2022). The behavior is ubiquitous but its internal mechanism is opaque. The paper asks where, mechanistically, refusal lives. Building on the linear representation hypothesis — that concepts are encoded as directions in activation space (Park et al. 2023b; Elhage et al. 2022) — and on activation steering (Panickssery et al. 2023; Turner et al. 2023; Zou et al. 2023a), it tests the strong claim that refusal across many models is governed by a **single one-dimensional subspace** of the residual stream. The motivation is dual: scientific (understand alignment internals) and safety (if one direction gates refusal, current safety fine-tuning is brittle).

## Core contribution and precise claims

1. **Existence of a single refusal direction.** For each of 13 open-source chat models (1.8B–72B params), one residual-stream direction `r` is *necessary and sufficient* for refusal: ablating it from all layers/positions stops refusal of harmful prompts; adding it induces refusal on harmless prompts.
2. **A white-box jailbreak via weight orthogonalization** — a rank-one, training-free weight edit that permanently removes the direction with negligible capability loss, costing under $5 of compute for a 70B model.
3. **Mechanistic account of adversarial suffixes**: a GCG suffix bypasses refusal by *suppressing* the expression of the refusal direction, hijacking the attention of the heads that write to it.
The authors frame the single-direction result as an *existence proof*, conceding their extraction heuristic is "likely not optimal."

## Method and mathematics

**Transformer setup.** A decoder-only transformer maps tokens `t ∈ V^n` to distributions. Let `x_i^(l) ∈ R^{d_model}` be the residual stream of token `i` at the start of layer `l`. Each layer adds attention then MLP contributions:

```
x̃_i^(l) = x_i^(l) + Attn^(l)(x_{1:i}^(l)),   x_i^(l+1) = x̃_i^(l) + MLP^(l)(x̃_i^(l))      (1)
```

Final logits = `Unembed(x_i^(L+1))`, softmaxed to `y_i`. `I` denotes the *post-instruction* token positions (chat-template tokens after the user instruction); analysis focuses there.

**Difference-in-means direction.** Two small datasets are built: `D_harmful` (sampled from ADVBENCH, MALICIOUSINSTRUCT, TDC2023, HARMBENCH) and `D_harmless` (from ALPACA), each split 128 train / 32 val. For every layer `l ∈ [L]` and position `i ∈ I`, compute mean activations over harmful (`μ`) and harmless (`ν`) train prompts:

```
μ_i^(l) = (1/|D_harmful^train|) Σ_t x_i^(l)(t),    ν_i^(l) = (1/|D_harmless^train|) Σ_t x_i^(l)(t)      (2)
```

The candidate direction is the difference `r_i^(l) = μ_i^(l) − ν_i^(l)`. Its *direction* points from harmless to harmful mean activation; its *magnitude* is the inter-class distance. This yields `|I| × L` candidates.

**Selecting one vector.** Each candidate is scored on the validation sets via an efficient *refusal metric* (App. B). Define a small set `R ⊆ V` of tokens that begin refusals (e.g. {"I"} for Gemma); the refusal probability is `P_refusal(p) = Σ_{t∈R} p_t` for next-token distribution `p`, and the metric is its log-odds:

```
refusal_metric(p) = logit(P_refusal(p)) = log(Σ_{t∈R} p_t) − log(Σ_{t∈V\R} p_t)      (6–9)
```

Selection (App. C.1) computes, per candidate: **bypass_score** (mean refusal_metric on `D_harmful^val` *under ablation* — lower = better bypass), **induce_score** (mean refusal_metric on `D_harmless^val` *under addition*), and **kl_score** (mean KL of last-token distributions on harmless prompts with vs. without ablation). It picks the minimum-bypass_score direction subject to induce_score > 0 (sufficient to induce refusal), kl_score < 0.1 (does not perturb benign behavior), and `l* < 0.8L` (avoid directions that merely block unembedding of refusal tokens). The chosen vector is `r` with unit form `r̂`.

**Interventions.** *Activation addition* shifts a harmless input toward refusal at layer `l`, all positions:

```
x^(l)' ← x^(l) + r^(l)      (3)
```

*Directional ablation* erases `r̂` from every activation (all layers, all positions, both `x_i^(l)` and `x̃_i^(l)`):

```
x' ← x − r̂ r̂ᵀ x      (4)
```

This zeroes the projection onto `r̂`, so the model can never represent the direction. The negative-direction subtraction `x^(l)' ← x^(l) − r^(l)` (17) also bypasses refusal but pushes harmless activations off-distribution (higher CE loss), so ablation is more "surgical."

**Weight orthogonalization (the jailbreak).** Equivalently, orthogonalize every matrix `W_out` that *writes* to the residual stream (embedding, positional embedding, attention-out, MLP-out, plus output biases) against `r̂`:

```
W_out' ← W_out − r̂ r̂ᵀ W_out      (5)
```

App. E proves equivalence to inference-time ablation: with `x_post = x_pre + W_out t`, ablating gives `x_post' = x_pre − r̂r̂ᵀx_pre + (W_out − r̂r̂ᵀW_out)t`; if ablation already held upstream then `r̂ᵀx_pre = 0`, leaving `x_pre + W_out' t`. This is a permanent, gradient-free rank-one edit needing only harmful *instructions* — no harmful completions.

**Evaluation metrics.** Greedy decoding, 512-token cap. **refusal_score** = 1 if a completion contains any of 12 refusal substrings ("I'm sorry", "I cannot", "As an AI", …); **safety_score** uses Llama Guard 2 (safe = 1, unsafe = 0). Two metrics are used because each is individually flawed (a completion can refuse-then-comply, or comply benignly without refusal phrasing). Standard error `SE = √(p(1−p)/n)`, n = 100.

**Adversarial-suffix analysis.** Using **direct feature attribution** (DFA): a component's contribution is its output projected onto `r̂`. For Qwen 1.8B Chat, a GCG suffix suppresses last-token cosine similarity with `r̂` down to harmless levels; the top-8 highest-DFA attention heads have their output-to-`r̂` suppressed because the suffix *hijacks their attention* from the instruction region to the suffix tokens.

## Experimental setup and headline results

**Models (Table 1):** Qwen Chat (1.8/7/14/72B, AFT), Yi Chat (6/34B, AFT), Gemma IT (2/7B, APO), Llama-2 Chat (7/13/70B, APO), Llama-3 Instruct (8/70B, APO) — 13 models. **Necessity (§3.1):** ablating `r̂` over JAILBREAKBENCH (100 harmful prompts, 10 categories) collapses refusal and safety scores across all models. **Sufficiency (§3.2):** adding `r` over 100 ALPACA prompts induces refusal of benign requests.

**Jailbreak vs. baselines (Table 2, HARMBENCH 159 standard behaviors, ASR):** ORTHO reaches 79.2% (Qwen 7B), 84.3% (Qwen 14B), 78.0% (Qwen 72B) — on par with per-prompt GCG (79.5 / 83.5%). Llama-2 ASR is system-prompt sensitive: Llama-2 7B is 22.6% *with* the default safety system prompt vs. 79.9% without; Qwen is insensitive (79.2 vs 74.8%). A 12-prompt sweep gives Llama-2 7B mean 30.0% (SD 23.3) vs Qwen 7B 76.7% (SD 5.9).

**Coherence (Tables 3, 8).** MMLU/ARC/GSM8K/WINOGRANDE move <1% on average; most orthogonalized scores lie within 99% CIs of baseline (exceptions: Qwen 7B, Yi 34B). TruthfulQA consistently drops (e.g. Llama-3 70B 59.5 vs 61.8, −2.3) because its categories (misinformation, conspiracies) border refusal territory. CE-loss (Table 9) shows ablation barely changes loss on The Pile / Alpaca, while activation-addition inflates on-distribution loss (e.g. Llama-3 8B 0.213 vs 0.441). **Direction selection (Table 5):** chosen layers sit at ~0.3–0.8 depth (e.g. Llama-3 8B at layer 12/32, position −5; Qwen 72B at 62/80). **Base models (App. J):** the refusal direction is *already present in base models* and "repurposed" during safety fine-tuning, not learned from scratch — directly relevant to us.

## Limitations and threats to validity

The authors flag: (1) generality is unproven for untested/larger/proprietary/future models; (2) the extraction heuristic is "likely not optimal" — an existence proof, not the best estimator; (3) the suffix analysis is a *single model, single suffix* (Qwen 1.8B); (4) coherence is hard to measure and every metric is flawed; (5) the semantic identity of the direction is unclear — it may encode "harm" rather than "refusal." Threats: string-matching refusal_score misses paraphrased refusals; Llama Guard 2 is fallible; difference-in-means assumes a linear single-direction structure that later work disputes (see Validation notes).

## Relevance to our project (ICCD-3K, Phase 1)

Our study asks where in the network post-training alignment (via any fine-tuning method: SFT, RLHF, DPO, RLVR/GRPO, ...) **selectively** reshapes cultural knowledge and whether that change is recoverable — for each cultural content type, a recoverable late **gate** or an unrecoverable mid-layer **rewrite** — using Indian cultural minimal pairs, where Indian culture is at once the controlled probe for this mechanistic question and a genuine subject the study cares about. This paper is a direct methodological template.

**Methods/metrics we borrow.** (a) **Difference-in-means** (Eqs. 2) is exactly our tool for extracting a per-concept "cultural direction" from clean vs. corrupted prefix pairs — cheap, training-free, causally potent. (b) The **log-odds / logit metric** (Eqs. 6–9) is the principled form of our planned per-item *log-odds difference* on the target-answer token; we should adopt the full log-odds, `log p − log(1−p)`, not raw probability, to separate extreme values and stabilize the paired t-tests over our 60 cells × 50 items. (c) **Directional ablation** (Eq. 4) and **activation addition** (Eq. 3) give us necessity/sufficiency tests for a cultural direction at each layer — the natural experiment for "gate vs. rewrite." (d) **DFA** (project component output onto the direction) lets us attribute *which layers/heads* carry cultural information and whether alignment suppresses late (gating) vs. shifts mid (rewriting). (e) Their **base-vs-chat cosine-similarity** experiment (App. J) is the single most transferable design: if our cultural direction is present in the base model and merely re-expressed late in the aligned model, that is *gating*; if mid-layer geometry is restructured, that is *rewriting*. We should replicate it verbatim on Indian-culture pairs, applying the same base-vs-aligned logic equally to every fine-tuning method's model pair.

**Design constraints it implies.** Intervene/measure at **post-instruction token positions** and report the **source layer/position** like their Table 5 — our 60 cells should record `(layer, position)` provenance. Use **train/val/eval disjointness** and filtering (they enforce pairwise-disjoint splits) so probe-set construction does not leak. Their tiny n (128/32) shows difference-in-means needs few items, validating our 50-item cells, but we should still power the paired t-tests and report `SE = √(p(1−p)/n)`-style intervals.

**Pitfalls it warns against.** (1) **Semantic ambiguity** — a "cultural direction" may really encode formality, language, or sentiment; we need control pairs to rule out confounds, exactly as they concede "refusal" may be "harm." (2) **Off-distribution artifacts** from activation addition (Fig. 22) — prefer ablation or measure CE/KL side-effects so a coherence collapse is not mistaken for a representational change. (3) **String-match brittleness** — our log-odds target metric must key on robust answer tokens, not surface phrases. (4) **Single-model over-claiming** — generalize across model families before claiming "Indian culture is gated." (5) The contested **single-direction assumption**: 2026 follow-up argues refusal is *not* one-dimensional, so we should test rank ≥ 1 cultural subspaces.

## Validation notes

- **Title** "Refusal in Language Models Is Mediated by a Single Direction" — *confirmed*. Source: https://arxiv.org/abs/2406.11717
- **arXiv id 2406.11717** (submitted 17 Jun 2024; rev. 30 Oct 2024) — *confirmed* (plan citation correct). Source: https://arxiv.org/abs/2406.11717
- **Authors** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (this order) — *confirmed*. Sources: https://arxiv.org/abs/2406.11717 ; https://neurips.cc/virtual/2024/poster/93566
- **Venue NeurIPS 2024 (poster), NeurIPS 37** — *confirmed*. Sources: https://neurips.cc/virtual/2024/poster/93566 ; https://openreview.net/forum?id=pH3XAQME6c
- **13 chat models, up to 72B** — *confirmed* (abstract + Table 1; verified vs. arXiv abstract).
- **Single-direction claim is contested by later work** — *confirmed* a rebuttal exists: "There Is More to Refusal in Large Language Models than a Single Direction." Source: https://arxiv.org/html/2602.02132v1 (treat as 2026 preprint; existence verified via search index, full peer-review status *unverifiable* at this date).
- **Code release** at github.com/andyrdt/refusal_direction — *confirmed*. Source: https://github.com/andyrdt/refusal_direction
- Quantitative figures (128/32 splits, 100/159 eval sizes, ASR/coherence tables, Eqs. 1–17, Table 5 layers) — *confirmed* against the paper markdown; not independently re-derivable from the web but internally consistent.

## Verification Log

Independent adversarial fact-check performed 2026-05-31 against the source paper markdown (`d:/Mech-Interp-Cultural/papers/RefusalDirection/RefusalDirection.md`) and online sources. **Verdict: accurate — no factual corrections required.**

**Bibliographic facts (online + source md):**
- Title "Refusal in Language Models Is Mediated by a Single Direction" — CONFIRMED (arXiv abstract page; paper md line 1).
- arXiv id **2406.11717** — CONFIRMED (https://arxiv.org/abs/2406.11717). Plan citation correct.
- Author list/order Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda — CONFIRMED (paper md lines 3–15). Note: OpenReview lists "Nina Rimsky" / "Oscar Balcells Obeso" — same individuals; the paper/arXiv form (Panickssery / Obeso) used in the analysis is correct, not an error.
- Venue NeurIPS 2024 **poster**, NeurIPS 37 — CONFIRMED (https://openreview.net/forum?id=pH3XAQME6c reports NeurIPS 2024, poster; neurips.cc/virtual/2024/poster/93566).
- 13 chat models, 1.8B–72B — CONFIRMED (abstract; Table 1, paper md lines 67–76).
- Code repo github.com/andyrdt/refusal_direction — CONFIRMED (not re-fetched this round; unchanged).

**Mathematics (cross-checked symbol-for-symbol against paper md):**
- Eq. 1 (residual stream), Eq. 2 (difference-in-means μ−ν), Eq. 3 (activation addition), Eq. 4 (directional ablation x − r̂r̂ᵀx), Eq. 5 (weight orthogonalization W − r̂r̂ᵀW), Eq. 17 (subtraction) — all CONFIRMED, exact form and symbols (md lines 53–55, 85, 99, 107, 162).
- Refusal metric Eqs. **6–9**: P_refusal(p)=Σ_{t∈R} p_t (6); refusal_metric=logit(P_refusal) (7) = log(P/(1−P)) (8) = log(Σ_{t∈R}p_t) − log(Σ_{t∈V\R}p_t) (9) — CONFIRMED verbatim (md lines 469–477).
- Selection criteria: min bypass_score s.t. induce_score>0, kl_score<0.1, l<0.8L — CONFIRMED (md lines 515–522).

**Load-bearing quantitative claims (verified against paper md):**
- Splits 128 train / 32 val; eval 100 JailbreakBench / 159 HarmBench standard behaviors — CONFIRMED (md 63, 141, 171, 450). JailbreakBench = 10 categories (md 439); HarmBench = 6 categories (md 441).
- Table 2 ASR: ORTHO Qwen 7B 79.2 (74.8 no-sys), 14B 84.3, 72B 78.0 (79.2 no-sys); GCG 79.5/83.5; Llama-2 7B 22.6 (79.9 no-sys) — CONFIRMED (md 180–185).
- 12-prompt system sweep: Llama-2 7B mean 30.0% / SD 23.3%; Qwen 7B mean 76.7% / SD 5.9% — CONFIRMED (md 677; Figure 19 lists 12 prompts).
- CE loss Llama-3 8B (on-distribution): baseline 0.195 / ablation 0.213 / act-add 0.441 — CONFIRMED (md 781, 847–849).
- TruthfulQA Llama-3 70B 59.5 vs 61.8 (−2.3) — CONFIRMED (md 212, 737).
- Table 5 source layers: Llama-3 8B l*=12/32, i*=−5; Qwen 72B l*=62/80 — CONFIRMED (md 553, 545).
- App. J base-model "repurposing" (high cosine sim on harmful prompts in base models) — CONFIRMED (md 864–872).

**Contested follow-up:**
- "There Is More to Refusal in Large Language Models than a Single Direction", arXiv **2602.02132** (Joad, Hawasly, Boughorbel, Durrani, Sencar; Feb 2026) — existence CONFIRMED via arXiv/ADS index. Full peer-review status remains UNVERIFIABLE at this date; the analysis already hedges this correctly.

**Minor notes (no edit made):**
- The analysis gives the Gemma refusal-token set as `R = {"I"}` (with "e.g."). Table 4 (md 498) lists Gemma's authoritative set as {235285} = "I cannot", while Figure 10's caption (md 491) uses token 234285 = 'I' for Gemma 2B specifically. The "e.g. {"I"}" framing is defensible given Figure 10; left unchanged.
- Word count: the analytical body (everything above this Verification Log) is **1834 words**, i.e. ~34 words (2.3%) over the requested 1500–1800 band. Flagged as marginally over; not padded or trimmed, since trimming load-bearing content to hit the band is not warranted for so small an overage.
