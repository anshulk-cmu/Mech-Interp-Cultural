# Mech-Interp-Cultural — The Selective Alignment Project

A mechanistic-interpretability study of post-training alignment: **does RLHF *rewrite* a model's mid-layer
representations of culturally specific content, or *gate* them with a late-stage policy that leaves the
representations intact?** Indian cultural knowledge is the *testbed*, not the topic. The instrument is
**ICCD-6K**, a 6,000-item controlled minimal-pair probe set (CounterFact/ROME lineage), purpose-built for
this measurement — deliberately bounded, not a comprehensive cultural benchmark.

Working title: *The Selectivity of RLHF: Mechanistic Sources of Cultural Flattening in Post-Training Alignment.*

---

## Repository layout

| Path | What it holds |
|---|---|
| [`papers/`](papers/) | The 7 source research papers as OCR markdown, one folder per paper. **Inputs.** |
| [`paper-analyses/`](paper-analyses/) | Our web-validated, fact-checked technical analysis of each paper (one file each). **Stage-1 output.** |
| [`plans/`](plans/) | Planning documents (see below). |
| [`plans/ICCD-6K-plan.md`](plans/ICCD-6K-plan.md) | **The current Phase 1 dataset-construction plan (v1.2).** ← start here for Phase 1. |
| [`plans/project-plan.md`](plans/project-plan.md) | End-to-end project plan across all 5 phases. |
| [`plans/_archive/`](plans/_archive/) | Superseded plan versions, kept for lineage (e.g. the v1.1-WIP ICCD-3K plan). |

> Navigation aid: `papers/` (the sources) ↔ `paper-analyses/` (what we made of them) ↔ `plans/` (where the project is going).

---

## The 7 papers and why each is here

| Folder | Paper (verified) | Role in this project |
|---|---|---|
| `ROME/` | Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022 (arXiv:2202.05262) | Minimal-pair / CounterFact template; mid-layer-MLP "rewriting" signature. |
| `IOI/` | Wang et al., *Interpretability in the Wild* (IOI), ICLR 2023 (arXiv:2211.00593) | Path patching; logit-difference metric; self-repair caveat. |
| `RefusalDirection/` | Arditi et al., *Refusal Is Mediated by a Single Direction*, NeurIPS 2024 (arXiv:2406.11717) | Axis C method; base-vs-aligned "repurposing" logic. |
| `ActivationPatching/` | Zhang & Nanda, *Towards Best Practices of Activation Patching*, ICLR 2024 (arXiv:2309.16042) | Governs corruption (STR over Gaussian noising) and metric (logit difference over probability). |
| `CLMBench/` | Hu et al., *CLM-Bench*, arXiv:2601.17397 (2026) | Closest prior work; layer-selection diagnostic; "generation collapse ≠ knowledge erasure." |
| `Indica/` | Madhusudan et al., *Common to Whom?* (INDICA), ACL 2026 Main (arXiv:2601.15550) | Validated 5-region taxonomy; LLM-judge over-merge warning. |
| `Sanskriti/` | Maji et al., *SANSKRITI*, ACL Findings 2025 (arXiv:2506.15355) | Cultural content source (21,853 items); answer-replacement label-noise caveat. |

---

## Status

- **Phase 1 plan:** v1.2 (ICCD-6K), current. Depth raised from 50→100 items/cell (3,000→6,000) **for statistical
  power only**; the 60-cell scope (3 axes × 5 regions × 4 sub-concepts) is fixed. The 3K configuration is retained
  as a documented fallback.
- Phases 2–5 will get their own detailed specs before each runs.
- Everything is version-tracked in git. Nothing is locked except the OSF pre-registration, filed before Stage 8.
