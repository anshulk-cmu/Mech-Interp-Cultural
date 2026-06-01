# ICCD-6K Phase 1 — Smoke-Test Report (A01-SS-01)

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

Date: 2026-05-31. Goal: prove the construction pipeline end-to-end by building **100 final
items** for one cell — **A01-SS-01** (Regional Specificity / South / Festivals) — before scaling
to all 60 cells.

> **Superseded by quality-hardening v2 (see `docs/QUALITY.md`).** This report describes the first
> proof-of-pipeline run (~70–75% clean, Kerala-dominated). The v2 run hardened it to **100% strict-clean,
> rebalanced across South states (Kerala 39 / TN 24 / Karnataka 16 / AP 14 / Telangana 7), 0 provenance
> gaps**. The pipeline/architecture description below still holds; the numbers are updated in QUALITY.md.

## Outcome

**100 final, cross-validation-complete items** at `data/final/iccd_A01-SS-01.json`.
The full chain ran: SANSKRITI + Wikipedia sourcing → STR cross-anchor-swap minimal pairs →
F1–F7 filters → fp16 Llama-3.1-8B delta-L gate on AWS → Claude 4-check verification (in-session
subagents) → token-balanced final selection → provenance audit (0 cross-validation gaps).

## Funnel

```
SANSKRITI cell items 323
  → Stage 2 candidates        174   (SANSKRITI anchors + recursive Wikipedia festival categories;
                                       kept if SANSKRITI-attested OR Wikipedia intro confirms the state)
  → Stage 3 STR pairs         172   (cross-anchor swap: corruptor = different-region festival, r' ≠ r)
  → Stage 4 filters kept      165   (hard F1/F5/F6/F7 + clear leaks; borderline flagged for Claude)
  → Stage 8 model-kept        163   (Llama-3.1-8B fp16, delta-L > 1.0 nat)
  → Claude-approved           108   (fact / leakage / counterfactual / naturalness)
  → Pass B final              100   (50/30/20 token balance, realized 61/17/22; seed 42)
```

## Where each stage ran

- **Local (`culture` conda env, no torch):** Stages 0–5, plus verdict merge + final assembly.
- **AWS g6.xlarge (L4 24 GB, fp16):** Stage 8 model scoring. Instance + SG provisioned, used, and
  torn down (~25 min, ≈ $0.35). No orphaned resources.
- **Claude Code subagents (Max subscription, no API key):** Tier-1.5 verification, 4 parallel batches.

## What the Claude tier caught (the point of using it)

The deterministic filters were deliberately lenient (your directive) so Claude could flexibly
adjudicate rather than brittle string-matching reject valid items. It flagged 55/163:
- **Not festivals** the recursive Wikipedia walk pulled in — temples, a monastery, a hockey
  tournament, a "Fruit Ganesh" shop (fact_ok false).
- **Place-name leakage** F2 only flagged — anchors containing Thrissur / Hyderabad / Chennai /
  Mysore / Bengaluru, which give away the state (leakage_ok false).
- **Malformed corruptors** from SANSKRITI corruptor extraction — mojibake "Tokhü Emong",
  "Ganesh Chaturthi in maharashtra", "Temple fairs (Jatras)" (natural_ok false).

The 108 approved all carry clean corruptors, so the final 100 is clean.

## Reproducibility

- Seeds: main 42, held-out 137, spot-check 314, counterfactual-audit 271.
- Tokenizer pinned: `meta-llama/Llama-3.1-8B` @ `d04e592b…`. Versions in `requirements-local.txt`.
- Every item: Wikipedia `oldid` + access timestamp, `cross_validated_by`, Stage-8 delta-L, Claude verdict.
- Wikipedia API responses cached (`data/resources/wiki_cache/`); all stages idempotent/resumable.

## Known issues to fix before scaling to 60 cells

1. **Corruptor extraction (Stage 3)** produces some malformed anchors → wasted candidates (Claude
   caught them). Clean the corruptor pool or source corruptors from Wikipedia, not raw SANSKRITI questions.
2. **F2 place-leak** should hard-reject capital/major-city names inside the anchor.
3. **Token balance** for Axis A is target=state-constrained (only Tamil Nadu is 2-token in the South),
   so the 50/30/20 quota uses fallback fill; this is inherent to Axis A and recorded per item.
4. **Web tier** (general-internet sourcing) is authorized and designed but was not needed here.

## Verdict

Pipeline validated end-to-end at 100-item scale, including real GPU model scoring and the Claude
verification layer. Scaling to 60 cells is a config change (loop cells) plus the corruptor cleanup above.
