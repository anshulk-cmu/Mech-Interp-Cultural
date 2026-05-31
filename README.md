# Mech-Interp-Cultural — The Selective Alignment Project

Does post-training alignment (RLHF) **rewrite** a model's mid-layer representations of culturally specific content, or **gate** them with a late-stage policy that leaves the representations intact? This project answers that mechanistically, with Indian cultural knowledge as the *testbed*, not the topic.

Working title: *The Selectivity of RLHF: Mechanistic Sources of Cultural Flattening in Post-Training Alignment.*

## The question

Aligned models behave differently from their base versions on culturally specific inputs, and not always for the better — refusals, indigenous terms flattened to generic ones, regional facts apparently lost. Two hypotheses explain this identically in behavior but differently inside the network: **representation rewriting** (mid-layer content is genuinely altered) versus **late-stage gating** — the mechanistic form of the Superficial Alignment Hypothesis, where knowledge persists and a late mechanism redirects the output. Behavior alone cannot separate them; mechanistic interpretability can. The distinction matters: a gate is recoverable by activation-level intervention, whereas rewriting requires retraining.

## Approach

We run **paired checkpoints** (a base model and its post-trained version) through a controlled probe set, then locate the change with layer-wise probing, sparse dictionary learning, and causal patching. Four pre-registered outcomes — A rewriting, B gating, C mixed-by-axis, D no signal — are each mapped *in advance* to a measurable per-phase signature (the decision-rule table in [plans/project-plan.md](plans/project-plan.md)).

## ICCD-6K — the probe set

A 6,000-item controlled minimal-pair probe set in the CounterFact/ROME lineage — purpose-built for this measurement, not a benchmark and not comprehensive cultural coverage. Fixed scope: **60 cells = 3 axes × 5 regions × 4 sub-concepts** (regions per INDICA). Depth is 100 items/cell, raised from 50 *purely for statistical power* (Holm-corrected per-cell detectable effect d≈0.59 → d≈0.42); breadth is unchanged, and the 3K/50 configuration is retained as a documented fallback. The per-item readout is a log-odds difference (ΔL, in nats) — never a raw probability, so answer-suppression stays detectable; Axis A adds a normalized logit difference LD(r,r′). Corruption is axis-specific Symmetric Token Replacement: Axis A a cross-anchor swap defining r′, Axes B/C description-based (C with reversed sign). QA is two-tier — automated first (deterministic filters + Wikipedia-and-SANSKRITI cross-validation + base-model validation), then human review.

## The five phases

1. **Dataset** — build ICCD-6K (full spec: [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md)).
2. **Layer-wise probing** — linear + MDL probes, Logit-Lens KL, Direct Logit Attribution.
3. **Dictionary learning** — BatchTopK Crosscoders (Δ-norm: preserved/shifted/exclusive) + Skip Transcoders, trained on background text.
4. **Causal validation** — cross-checkpoint patching, path patching, refusal-direction ablation/addition, latent-feature steering.
5. **Synthesis** — pre-registered per-axis verdict plus cross-model generality.

**Model suite:** Pair 1 Llama-3-8B (base/Instruct) and Pair 2 Gemma-2-9B (base/it) are the clean Western-RLHF core; Pair 3 Mistral-Small-3.1-24B / Sarvam-M is exploratory (SFT+RLVR, multi-factor confound).

## Repository

| Path | Contents |
|---|---|
| [papers/](papers/) | 7 source papers (OCR markdown), one folder each |
| [paper-analyses/](paper-analyses/) | our web-validated technical analysis of each |
| [plans/project-plan.md](plans/project-plan.md) | end-to-end plan (all five phases) |
| [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md) | Phase 1 dataset spec |
| [plans/_archive/](plans/_archive/) | superseded versions, kept for lineage |

**Papers:** ROME (arXiv:2202.05262), IOI (2211.00593), Refusal Direction (2406.11717), Activation Patching (2309.16042), CLM-Bench (2601.17397), INDICA (2601.15550), SANSKRITI (2506.15355).

## Status

v1.2 (ICCD-6K). Reproducible (pinned seeds and commit hashes, archived Wikipedia snapshot), pre-registered on OSF before model-side validation; CC-BY-4.0 data / MIT code. Start at [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md) for Phase 1, or [plans/project-plan.md](plans/project-plan.md) for the whole study.
