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

---

## A01-NN-01 (North) + A01-EE-01 (East) — Tier-1.5 quality (pre-Stage-8)

Built to the same standard as A01-SS-01. Because the Stage-8 gate needs AWS/Babel, **Tier-1.5 Claude
verification was run before the gate** on the full Stage-4 set, so the verified pool is the
construct-validity-clean batch that goes to scoring. Verification was multi-agent with **live web
fact-checking** — every festival→state attribution confirmed against a real source, nothing assumed
or manufactured.

| Metric | A01-NN-01 | A01-EE-01 |
|---|---|---|
| Stage-4 filtered | 155 | 168 |
| Tier-1.5 verified pass | **126** | **121** |
| Provenance gaps | 0 (90 Wikipedia oldid + 36 web URL) | 0 (112 Wikipedia oldid + 9 web URL) |
| State-name leaks | 0 | 0 |
| Bad counterfactuals (r′=r / same-region) | 0 | 0 |
| Duplicate anchors (incl. fuzzy variants) | 0 | 0 |
| States represented | 7 (Raj 33 / Ukd 25 / UP 23 / Pun 17 / Lad 14 / Har 11 / Del 3) | 10 (Od 19 / WB 18 / As 18 / Nag 14 / Tri 11 / Meg 9 / Bih 9 / Man 9 / Sik 7 / Miz 7) |
| Token 1/2/3 | 53/48/25 | 9/96/16 |

**Why some states are absent** — the Axis-A target *is* the state, and the F1 cap rejects targets
>3 tokens. So **Jammu & Kashmir, Himachal Pradesh, Chandigarh (North) and Arunachal Pradesh,
Jharkhand (East) are excluded by construction**, not by lack of sourcing. East's 2-token-heavy
token mix is the same Axis-A state↔token correlation noted for South (only Tamil Nadu was 2-token
there); the realized distribution is recorded and Pass B fills from strata per the Stage-5 fallback.

**What Tier-1.5 caught (the point of the gate)** — across the two cells it rejected, with sources:
non-festivals the category/web sourcing pulled in — a river (*Brahmaputra*), an island (*Majuli*),
a potters' quarter (*Kumortuli*), deities (*Durga*, *Jagaddhatri*, *Dharmathakur*), a GI-tagged cloth
(*Gamosa*), a radio programme (*Mahisasuramardini*), films (*Delhi-6*, *Raktabeej*), a personal name
(*Ananya*), organisations/clubs (*Bullygunge Cultural Association*, *Barowari*), college/tech and
literary fests (*Abeyaantrix*, *Bharat Rang Mahotsav*, *Bundelkhand Literature*); modern
tourism-promotional "Mahotsav/Utsav" branding (*Sangai*, *Shirui Lily*, *Dwijing*, *Namami Brahmaputra*,
*Kaziranga Elephant*, *Konark Dance Festival*, *Rajgir/Patna/Basti Mahotsav*, *Bundi Utsav*); pan-Indian
non-distinctive festivals (*Makar Sankranti*, *Krishna Janmashtami*, *Durga Ashtami*, bare *Rath Yatra*);
and **mis-attributions** (*Bundeli Utsav*→Madhya Pradesh not UP; *Lai Haraoba*→Manipur not Assam;
*Jagaddhatri*→West Bengal not Odisha). All four checks (fact/leakage/counterfactual/naturalness) were
applied; counterfactual_ok was 100% (the curated corruptor bank is clean).

**Status: RELEASED.** Both cells were scored on the full 6-model suite (4 small on AWS g6.xlarge L4,
2×24B on CMU Babel A6000×4, run in parallel; AWS instance + SG torn down, Babel wiped). Base-Llama-3.1-8B
gate (ΔL>1.0): **North 126→119, East 121→112** — both clear 100 with margin. Token-balanced Pass B
(seed 42) drew the final **100 each** (North token mix hit 50/30/20 exactly; East 7/77/16, the recorded
Axis-A state↔token correlation), with **0 missing cross-validation**. `data/final/iccd_{A01-NN-01,A01-EE-01}.json`;
manifest now 3/60 cells, 300 items. Cross-model `model_scores` + `cross_model_delta` appended for all 6 models
(see `docs/CROSSMODEL.md`): clean-RLHF Llama/Gemma preserve or sharpen the binding (corr 0.87–0.94), while
the Indian-aligned **Sarvam-M weakens it (corr 0.60 North / 0.50 East; Δ +0.53 / +0.96)** — the same
rewrite-direction signal as A01-SS-01, and stronger in East than North (an early regional-selectivity hint;
single confounded arm, directional only — not a rewrite-vs-gate verdict, which is Phase 4).

---

## A01-WW-01 (West) + A01-CC-01 (Central) — Tier-1.5 quality (the festival-thin regions)

The two SANSKRITI-thin regions (West 132 / Central 104), so the candidate pool came mostly from the
**web tier** — a structured-output web-research Workflow with live fact-checking, then a Tier-1.5
verify Workflow, then a fuzzy-dup pass (`scripts/apply_verdicts.py`). Every festival→state attribution
is web-verified with a logged source URL; nothing assumed or manufactured.

| Metric | A01-WW-01 | A01-CC-01 |
|---|---|---|
| Stage-4 filtered | 143 | 134 |
| Tier-1.5 verified pass | 121 | 118 |
| After fuzzy-dup drop | **111** (−10) | **113** (−5) |
| Provenance gaps | 0 (72 Wikipedia oldid + 28 web URL) | 0 (54 Wikipedia oldid + 46 web URL) |
| State-name leaks | 0 | 0 |
| Bad counterfactuals (r′=r / same-region) | 0 | 0 |
| States represented | 3 (Mah 40 / Guj 30 / Goa 30) | 2 (MP 58 / Chh 42) |
| Token mix | 1-tok 100 | 3-tok 58 / **5-tok 42** |

**Why the token mixes are extreme** — Axis-A targets are state names. All three eligible West states
(Gujarat, Maharashtra, Goa) are **single-token**, so West is a clean 1-token stratum; Dadra & Nagar
Haveli & Daman & Diu (13 tok) is excluded by the F1 cap. Central has only **Madhya Pradesh (3 tok)**
and **Chhattisgarh** — which tokenizes to **5**, over the cap. Rather than collapse Central to one
state, Chhattisgarh is kept as a **documented F1 exception** (`config.F1_TOKEN_EXCEPTIONS`), forming a
5-token stratum that Pass B's fallback fill carries (user-approved 2026-05-31). Within-cell token
length is therefore constant-or-near-constant — which *controls* the confound by construction; these
two cells simply don't contribute cross-token-length variation.

**What Tier-1.5 caught** — modern govt "Mahotsav/Utsav" tourism brandings (*Bhojpur Utsav*,
*Bhoramdeo Mahotsav*), district-name leakage (anchors embedding *Bastar* → Chhattisgarh), EDM/arts
events (*Sunburn*, *Serendipity*), non-festivals, and fuzzy spelling/qualifier variants dropped by
`apply_verdicts.py` (*Zatra/Jatra*, *Madhavpur Ghed/Fair*, *Haji Pir/Hajipir*, *Akti/Akti Tihar*).
counterfactual_ok was 100% (clean corruptor bank).

**Status: RELEASED.** Scored on the full 6-model suite (combined `suite_input_WWCC.json`, 224 items;
4 small on AWS g6.xlarge L4, 2×24B on CMU Babel A6000×4 job 8227432, run in parallel; AWS instance + SG
torn down, Babel token/dir/cache wiped). Base-Llama gate (ΔL>1.0): **West 111→105, Central 113→102** —
both clear 100 with margin. Token-balanced Pass B (seed 42) drew the final **100 each**, with **0
missing cross-validation**. `data/final/iccd_{A01-WW-01,A01-CC-01}.json`; manifest now **5/60 cells,
500 items**. Cross-model `model_scores` + `cross_model_delta` appended for all 6 models: clean-RLHF
Llama/Gemma preserve or sharpen the binding (corr Llama 0.89 West / 0.86 Central; Gemma 0.91/0.92,
aligned *sharper*), while the Indian-aligned **Sarvam-M weakens it most (corr 0.61 West / 0.50
Central; Δ +0.29 / +0.03)** — Central ties East for the weakest correlation, reinforcing the
regional-selectivity hint (directional, single confounded arm — Phase 4 adjudicates).
