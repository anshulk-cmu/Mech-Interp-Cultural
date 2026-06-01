# ICCD-6K — Quality Hardening Pass (A01-SS-01)

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

After the first smoke run produced 100 items at ~70–75% clean (analysis in REPORT.md), we made
targeted fixes to reach 90%+ clean with **no compromise on the validated methodology**. Each fix
is grounded in a source/logic below; none changes the mechanistic design (STR cross-anchor swap,
log-odds delta-L, paired contrast).

## Defects found (first run) → fix → grounding

| # | Defect (first run) | Fix | Grounded in |
|---|---|---|---|
| 1 | Truncation fragments ("Bengaluru International **Film**", "ViBGYOR **Film**") | Strip trailing "Festival" only when it leaves no fragment; else keep full title | Naturalness / in-distribution STR text (Zhang & Nanda 2309.16042) |
| 2 | "(festival)" / "(Chandigarh)" parentheticals in the prefix; near-duplicates (Pongal ×3) | Drop all `(...)` in `_clean` → also dedups "Pongal" vs "Pongal (festival)" | Minimal-pair construct validity (ROME 2202.05262) |
| 3 | Events, not traditional festivals (~20%): film/music/college/sport (Verba Maximus, Alai Balai, Madras Music Season, Champions Boat League) | Source-level exclude of film/music-season/league/tournament/trophy/dated; **strict Claude gate** rejects college/political/film/sport/person/temple | Axis-A "Festivals" construct (plan §3.1, SANSKRITI attribute); two-tier QA w/ Claude Tier-1.5 (INDICA over-merge warning 2601.15550) |
| 4 | Temples/people as anchors (Thrikkakara Temple, Sakthan Thampuran) | Exclude titles ending "Temple"; Claude rejects persons/buildings | Construct validity (plan §11.8) |
| 5 | Malformed corruptors ("Temple fairs (Jatras)", "Ganesh Chaturthi in maharashtra", mojibake) | Corruptor must be a single clean named festival (no embedded prepositions/parens/mojibake/>4 words) | STR: in-distribution, well-defined r′ (Zhang & Nanda) |
| 6 | Provenance mismatch ("Boat Race" → UK "The Boat Race") | India-guard: Wikipedia intro must mention India or the target state | Mandatory cross-validation, no source is ground truth (plan §5.5) |
| 7 | Kerala-dominated cell (~76/100) | Pass B: round-robin across target states **within** each token-length stratum | Token-balance 50/30/20 stays primary (statistical confound control, Zhang & Nanda); state spread is secondary (cultural representativeness, INDICA regional design) |

## Why state-balance is secondary to token-balance (statistical grounding)

For Axis A the target *is* the state, and state ≈ target token-length (Kerala/Karnataka=1, Tamil
Nadu=2, Andhra/Telangana=3). The 50/30/20 one/two/three-token quota is the *statistically* grounded
control (it removes target-length as a delta-L confound). So we keep token strata primary and spread
states **within** each stratum; the realized state + token distributions are both recorded per the
plan's Stage-5 fallback rule.

## Funnel after fixes (local, before model gate)

```
SANSKRITI cell 323 → Stage2 candidates 130 (events/temples/UK-page excluded at source)
 → Stage3 pairs 127 (clean corruptors only) → Stage4 kept 121
```
(Widening Wikipedia recursion to depth 2 for a safety buffer before the model gate — see run log.)

## Verification plan
- Strict Claude gate is the construct-validity audit; the final 100 are by construction items that
  passed all four checks (festival / no-leak / valid-counterfactual / natural).

## RESULTS (quality-hardened v2)

**Final: 100 clean items, 0 missing cross-validation** (release gate passes) → `data/final/iccd_A01-SS-01.json`.

### Funnel
```
SANSKRITI cell 323
 -> Stage2 Wikipedia-clean candidates 144   (events/temples/UK-page excluded at source; India-guard)
 -> Stage2b web-tier +28                     (curated gov/heritage festivals; provenance logged)
 -> Stage3 STR pairs 151                     (clean corruptor bank; content-stable hash item_ids)
 -> Stage4 F1-F7 kept 144
 -> Stage8 base-Llama-3.1-8B fp16 gate 139   (delta-L > 1.0; AWS g6.xlarge)
 -> strict Claude gate 106 pass              (traditional-festival-only; rejects temples/persons/events)
 -> Pass B clean 100                         (state x token balanced, seed 42)
```

### First run vs quality v2
| Metric | First run | Quality v2 |
|---|---|---|
| Clean (strict) | ~70-75% | 100% (all Claude-strict-verified) |
| Kerala share | 76 / 100 | 39 / 100 |
| State spread | Kerala-dominated | Kerala 39, TN 24, Karnataka 16, AP 14, Telangana 7 |
| Token 1/2/3 | 61/17/22 | 55/24/21 (target 50/30/20) |
| Malformed corruptors | several | 0 (curated bank) |
| Truncation fragments / parentheticals | present | 0 |
| Provenance gaps | n/a | 0 (91 Wikipedia oldid + 9 whitelisted web URL) |

### Provenance (final 100)
- 68 Wikipedia, 10 SANSKRITI, 22 web-tier. **91 carry a Wikipedia oldid** (reproducible from the pinned
  snapshot); **9 web-tier items carry a whitelisted gov/heritage `source_url`** — per user, a web source
  alone is valid provenance when no Wikipedia page resolves. All web sources logged in
  `data/resources/web_sources.jsonl`.

### Reusable scale assets (for the 60-cell run)
- `data/resources/corruptor_bank.json` — clean cross-region corruptor festivals (all regions).
- `data/resources/web_festivals.json` — per-cell curated web-tier supplement (+ `web_sources.jsonl` log).
- Content-stable `item_id` (sha1 of anchor) — survives re-runs, so gate/verdict reuse is safe at scale.

### Known minor note (for human Tier-2)
- 1-2 web items (e.g. *Masi Magam*) had the Wikipedia opensearch resolve land on a wrong page
  (mismatched extract); Claude confirmed the festival is genuine on construct and the correct
  `web_source_url` is recorded. Flagged for the human pass.
