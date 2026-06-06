# Mech-Interp-Cultural — The Selective Alignment Project

Post-training alignment (via any fine-tuning method: SFT, RLHF, DPO, RLVR/GRPO, ...) reshapes a model's culturally specific knowledge **selectively** — sparing some content types and regions while degrading others. That non-uniformity is documented behaviorally; what is open is **where in the network it lives and whether it is recoverable**. For each kind of cultural content we test whether alignment installs a recoverable late **gate** (mid-layer representation intact, output redirected) or an unrecoverable mid-layer **rewrite**, answered mechanistically with a paired base-vs-aligned design — with Indian cultural knowledge as both the controlled *testbed* for the mechanistic question and a genuine subject of study in its own right.

Working title: *The Selectivity of Post-Training Alignment: Mechanistic Sources of Cultural Flattening in Aligned Language Models.*

**Author:** Anshul Kumar, Carnegie Mellon University — [anshulk@andrew.cmu.edu](mailto:anshulk@andrew.cmu.edu)

## The question

Aligned models behave differently from their base versions on culturally specific inputs, and not always for the better — refusals, indigenous terms flattened to generic ones, regional facts apparently lost. Crucially the degradation is **selective**: it falls on some content types and regions more than others, and the region-level non-uniformity is already documented behaviorally (INDICA, arXiv:2601.15550). That selectivity is the starting point, not the finding. Any single such change has two explanations, identical in behavior but different inside the network: **representation rewriting** (mid-layer content is genuinely altered) versus **late-stage gating** — the mechanistic form of the Superficial Alignment Hypothesis, where knowledge persists and a late mechanism redirects the output. Behavior alone cannot separate them; mechanistic interpretability can. This is the discriminator we test: *if* a change is gating, the underlying knowledge is intact and in principle recoverable by an activation-level intervention; *if* rewriting, it is altered and needs retraining. The contribution is to **locate that selectivity in the network and adjudicate, per content type, which mechanism is engaged.**

## Approach

We run **paired checkpoints** (a base model and its post-trained version) through a controlled probe set, then locate the change with layer-wise probing, sparse dictionary learning, and causal patching. The adjudicator is a pre-registered **Recovery Test**: a *gate* is affirmed when a late-locus intervention (refusal-direction ablation, or suppression of a late alignment-exclusive feature) restores the target while mid-layer probes show the content was intact; a *rewrite* is carried by positive mid-layer signatures (probe drop, accessibility loss, mid-layer patch peak). A non-recovery is reported as unresolved, never as erasure; interchange patching is within-family only. The four pre-registered outcomes — A rewriting, B gating, **C mixed-by-axis (the selectivity result)**, D no signal — are each mapped *in advance* to a measurable per-phase signature (the decision-rule table in [plans/project-plan.md](plans/project-plan.md)). Whether the by-axis mechanism pattern is itself non-uniform — selectivity proper — is a pre-registered test, not read post hoc. The same forward passes yield a **behavioral bridge**: a 2-alternative `LD(r,r′)` accuracy and its base→aligned change, connecting the internal mechanism to observable right/wrong behavior.

## ICCD-6K — the probe set

A 6,000-item controlled minimal-pair probe set in the CounterFact/ROME lineage — purpose-built for this measurement, not a benchmark and not comprehensive cultural coverage. Fixed scope: **60 cells = 3 axes × 5 regions × 4 sub-concepts** (regions per INDICA). Depth is 100 items/cell, raised from 50 *purely for statistical power* (Holm-corrected per-cell detectable effect d≈0.59 → d≈0.42); breadth is unchanged, and the 3K/50 configuration is retained as a documented fallback. The per-item readout is a log-odds difference (ΔL, in nats) — never a raw probability, so answer-suppression stays detectable; Axis A adds a normalized logit difference LD(r,r′). Corruption is axis-specific Symmetric Token Replacement: Axis A a cross-anchor swap defining r′, Axes B/C description-based (C with reversed sign). QA is two-tier — automated first (deterministic filters + Wikipedia-and-SANSKRITI cross-validation + base-model validation), then human review.

## The five phases

1. **Dataset** — build ICCD-6K (full spec: [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md)).
2. **Layer-wise probing** — linear + MDL probes, Logit-Lens KL, Direct Logit Attribution.
3. **Dictionary learning** — BatchTopK Crosscoders (Δ-norm: preserved/shifted/exclusive) + Skip Transcoders, trained on background text.
4. **Causal validation — the Recovery Test** — cross-checkpoint patching (within-family), path patching, refusal-direction ablation/addition, latent-feature steering; recoverability locus + direction adjudicates gating (rewrite is carried by mid-layer signatures).
5. **Synthesis** — pre-registered per-axis verdict plus cross-model generality.

**Model suite:** three co-equal model arms, each covering a fine-tuning method and all primary. Pair 1 Llama-3-8B (base/Instruct) and Pair 2 Gemma-2-9B (base/it) cover human-preference RLHF; Pair 3 Mistral-Small-3.1-24B / Sarvam-M covers SFT+RLVR via GRPO. The base-vs-aligned logic and statistical machinery apply equally to every arm.

## Repository

| Path | Contents |
|---|---|
| [plans/project-plan.md](plans/project-plan.md) | end-to-end plan (all five phases) |
| [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md) | Phase 1 dataset spec |
| [plans/_archive/](plans/_archive/) | superseded plan versions, kept for lineage |
| [iccd/](iccd/) | Phase-1 build pipeline — config, Wikipedia client, stages 0–6 + Claude verify |
| [scripts/](scripts/) | model scoring (`scoring/`), cross-model analysis, dev utilities |
| [docs/](docs/) | living build docs — setup, pipeline, QA, report, cross-model, 60-cell tracker |
| [data/](data/) | `final/` releases (tracked); `raw/`, `interim/`, `resources/` gitignored working data |
| [papers/](papers/) | 7 source papers (OCR markdown), one folder each |
| [paper-analyses/](paper-analyses/) | our web-validated technical analysis of each |

**Papers:** ROME (arXiv:2202.05262), IOI (2211.00593), Refusal Direction (2406.11717), Activation Patching (2309.16042), CLM-Bench (2601.17397), INDICA (2601.15550), SANSKRITI (2506.15355).

## Status

v1.3 (ICCD-6K, selectivity-led). Reproducible (pinned seeds and commit hashes, archived Wikipedia snapshot), pre-registered on OSF before model-side validation; CC-BY-4.0 data / MIT code. Start at [plans/ICCD-6K-plan.md](plans/ICCD-6K-plan.md) for Phase 1, or [plans/project-plan.md](plans/project-plan.md) for the whole study.

**Build progress (Phase 1): Axis A01 — Regional Specificity — is COMPLETE — 20/60 cells, 1,778 items, all six models scored.** All four sub-concept rows are done: Festivals ×5, Costume & Textile ×5, Cuisine ×5, Rituals & Ceremonies ×5. Sixteen cells reach the full 100; four release as documented short cells at their true few-eligible-state ceilings (A01-WW-02 textile 64, A01-CC-02 textile 41, A01-CC-03 cuisine 30, A01-CC-04 rituals 43). Each released item carries clean/corrupted prefixes + target, the base-Llama ΔL, the Tier-1.5 verdict, provenance (Wikipedia oldid / source URL), and all six models' scores + base→aligned cross-model deltas. Next: Axis A02 (Cultural Flattening). See [docs/CELLS.md](docs/CELLS.md) for the per-cell tracker.
