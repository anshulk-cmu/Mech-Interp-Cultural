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

## A01-NN-02 (North) + A01-SS-02 (South) — Tier-1.5 quality (first Costume & Textile cells)

First non-Festivals sub-concept; the pipeline was generalized by sub-concept (STEP 0, `docs/PIPELINE.md`)
before building. The A01-02 relation frame is `"{anchor} is a folk textile tradition associated with
the Indian state of"`; corruptors are same-sub-concept, cross-region textiles (`corruptor_bank.json`
`by_subconcept["02"]`); `fact_ok` requires a real, distinctive, GI-tag/handloom-board-documented
regional textile of the target state.

| Tier-1.5 rejection class (caught, costume-specific) | example | check |
|---|---|---|
| Fashion brand / retailer / designer | Fabindia, Anokhi, JJ Valaya, Tarun Tahiliani, Obeetee | fact_ok |
| Weaver CASTE / community name (not a textile) | Adaviyar, Basor, Bhuiyar, Koshta, Tanti | fact_ok |
| Jewellery / ornament / footwear | Bangle, Araimudi, Kunukku, Tilla Jutti | fact_ok |
| Painting on cloth (not a weave/embroidery) | Aipan, Pichwai, Phad, Cheriyal scroll | fact_ok |
| Government powerloom scheme / cluster (no GI) | Bathukamma saree, Sircilla "matchbox" saree | fact_ok |
| Pan-Indian non-distinctive garment | angavastra, Mundu, Wedding sari, Bandhani→TN (mis-attribution) | fact_ok |
| Bare place name, no textile noun | Bandar Polavaram, Bobbili | natural_ok / fact_ok |

Funnels: **A01-NN-02** 146 F1-F7 → 123 Tier-1.5 → 110 (−13 fuzzy-dup) → 101 base-Llama ΔL>1.0 gate
→ **100** (UP 28 / Rajasthan 26 / Punjab 21 / Uttarakhand 15 / Ladakh 8 / Haryana 2; token 47/38/15).
**A01-SS-02** 142 F1-F7 → 127 Tier-1.5 → 107 (−20 fuzzy-dup) → 106 gate / 104 survivors → **100**
(Karnataka 25 / TN 24 / AP 19 / Kerala 16 / Telangana 16; token 41/24/35; reached 100 via a 9-item
second scoring round). Both **0 missing cross-validation**; manifest now **7/60 cells, 700 items**.
The base-Llama ΔL gate kept textiles at **92 % (N) / 96 % (S)** — as strong as Festivals, because
Tier-1.5 pre-filters to documented GI textiles. `counterfactual_ok` ≈ 100 % (curated textile corruptor
bank). Fuzzy-dedup is the dominant attrition (saree/sari/silk name-variants).

Cross-model (all 6 models, appended to the release JSONs): clean-RLHF preserves the binding — Llama-3.1-8B
corr **0.94 N / 0.95 S**; Gemma-2-9B aligned is **sharper** (Δ −2.56 / −2.37, corr 0.91 / 0.94). The
Indian-aligned **Mistral→Sarvam-M weakens the textile→state binding the MOST of any cell to date —
corr 0.44 (North) / 0.35 (South), Δ +3.32 / +3.91, 79–84 % of items base>aligned.** South costume is
the strongest weakening yet, extending the Sarvam-M regional-selectivity signal from Festivals to a
second sub-concept (directional, single confounded arm — Phase 4 adjudicates).

## A01-EE-02 (East) — Tier-1.5 quality (released); A01-WW-02 (West) + A01-CC-02 (Central) — released short cells (wave 6)

**A01-EE-02 (East), released.** Same generalized chain; web-tier-dominated across all 10 eligible East
states. The **verify Workflow was hardened this wave** (judge from the provided evidence + own knowledge,
≤5 web searches, mandatory StructuredOutput call, batch 7) after schema-forced subagents repeatedly
over-researched and failed to emit (0/14 → 100 % emit). Funnel: 163 pairs → 156 F1-F7 → **131 Tier-1.5
pass** → 113 (−18 fuzzy-dup) → **108 base-Llama ΔL>1.0 gate (96 %)** → **100 final**.

| Metric | A01-EE-02 |
|---|---|
| Stage-4 filtered | 156 |
| Tier-1.5 verified pass | 131 |
| After fuzzy-dup drop | **113** (−18) |
| Base-Llama gate (ΔL>1.0) | **108** (96 %) |
| Final | **100** |
| Provenance gaps | 0 |
| States represented | 10 (Od 16 / As 16 / WB 14 / Man 12 / Miz 9 / Meg 9 / Bih 8 / Nag 7 / Tri 6 / Sik 3) |
| Token 1/2/3 | 8 / 80 / 12 (2-tok-heavy, the East Axis-A state↔token correlation) |

What Tier-1.5 caught: a weaver CASTE (*Bhulia*), a fashion BRAND (*Boito*), and a region-name **leak**
(*Bengal Batik* — the anchor literally contains "Bengal"); all four checks applied, counterfactual_ok ≈ 100 %.
Cross-model: Llama corr **0.92** (Δ +0.85), Gemma aligned **sharper** (corr 0.92, Δ −2.31), **Mistral→Sarvam-M
corr 0.47, Δ +4.28, 88/100 base>aligned — the strongest costume weakening of any cell** (NN +3.32 → SS +3.91 → EE +4.28).

**A01-WW-02 (West) + A01-CC-02 (Central) — RELEASED as documented short cells.** Both are few-eligible-state
Costume cells (West 3 states, Central 2) where distinctive documented textiles are finite, so 100 is
structurally unreachable — NOT a quality failure. Before releasing, an **exhaustive recalibration** ruled
out a sourcing-effort explanation: an LLM web sweep (15 agents; long-tail + GI/handloom inventories) plus a
deterministic **deep-Wikipedia category crawl** (`scripts/dev/wiki_deep_ceiling.py`; 872 articles, depth-3,
full-text state resolution) **converged** — Wikipedia surfaced zero new valid textiles, only
people/firms/museums/castes/wrong-state noise. Also verified that relaxing the F1 cap to ≤5 tokens adds NO
eligible states (West's only over-cap state is the 13-token UT; Chhattisgarh, 5-tok, is already admitted).

| Metric | A01-WW-02 (West) | A01-CC-02 (Central) |
|---|---|---|
| Stage-4 filtered | 114 | 53 |
| Tier-1.5 verified pass | 87 | 46 |
| After fuzzy-dup drop | 77 (−10) | 43 (−3) |
| Base-Llama gate (ΔL>1.0) | **64** (83 %) | **41** (95 %) |
| Final (released at real n) | **64** | **41** |
| States | Gujarat 43 / Maharashtra 18 / Goa 3 | Madhya Pradesh 22 / Chhattisgarh 19 |
| Token mix | 1-tok 64 (clean single-token stratum) | 3-tok 22 / 5-tok 19 (Chhattisgarh exception) |
| Provenance gaps | 0 | 0 |

The gate retained far more than the conservative `model_known` projection (West ~47→**64**, Central ~12→**41**)
— the base model binds most of these textiles to their state. What Tier-1.5 caught: weaver CASTE names
(Balai, Basor, Chik Baraik, Dhedh), a museum (Calico Museum), a Tajik mis-attribution (Chakan), Bhopal-capital
**leakage**, and pan-Indian non-distinctive garments (Angavastra, Choli, Ghagra choli, Ghoonghat). **Note
(Central):** the maximized pool keeps a few town/tribe-variant names of shared crafts (Nandana/Tarapur dabu;
the Bastar aal-dyed cotton family) — kept per the maximize-and-document decision; a human Tier-2 may collapse them.

**Cross-model (6/6, scored on ONE g6.12xlarge / 4×L4, all models consolidated):** clean-RLHF preserves binding —
Llama-3.1-8B corr **0.93** (West) / **0.82** (Central); **Gemma-2-9B aligned sharper** (Δ −2.56 / −2.57).
**Mistral→Sarvam-M (Indian SFT+RLVR) weakens binding most — Δ +4.48 (West) / +4.18 (Central), corr 0.19 / 0.43**,
completing the costume-axis Sarvam-M selectivity gradient across the whole row: NN +3.32 → SS +3.91 → EE +4.28 → WW +4.48 / CC +4.18.

## A01-NN-03 (North) + A01-SS-03 (South) — Cuisine, Tier-1.5 quality (released, wave 7)

First Cuisine cells, first row under the **permanent ≤5-token cap** (plan §13 v1.4: `MAX_TARGET_TOKENS` 3→5, balance `{1:30,2:25,3:20,4:15,5:10}`, exceptions emptied) — which made **J&K & Himachal Pradesh** eligible in NN-03 and **Puducherry** in SS-03. Both data-rich → both released at the full **100**.

| Metric | A01-NN-03 | A01-SS-03 |
|---|---|---|
| Candidates (SANSKRITI / Wiki / web) | 187 (13 / 47 / 127) | 207 (12 / **154** / 41) |
| Stage-4 filtered | 178 | 197 |
| Tier-1.5 verified pass | 143 | 134 |
| After fuzzy-dup drop | 139 (−4) | 129 (−5) |
| Base-Llama gate (ΔL>1.0) | **124** (89 %) | **128** (99 %) |
| Final | **100** | **100** |
| Provenance gaps | 0 | 0 |
| States | 9 (Raj 15 / Pun 13 / UP 12 / Utk 11 / J&K 11 / Lad 10 / HP 10 / Del 9 / Har 9) | 6 (Ker 21 / Kar 21 / TN 20 / AP 16 / Tel 16 / Pud 6) |
| Token 1/2/3/4 | 37 / 31 / 11 / 21 | 42 / 20 / 32 / 6 |

**Verify hardening (recovery).** Running BOTH cells' verify Workflows at once triggered the known schema-forced turn-budget emit-failure at scale (~75 % of batches completed with no `StructuredOutput` call → only 42/178 and 63/197 verdicts). Fixed by: (a) a stronger emit rule ("judge from knowledge + the wiki_extract, web-search ≤2 genuinely unfamiliar items, emit FIRST"); (b) a new `gen_verify_workflow.py --missing <verdicts.json>` redo mode (batch 5) that re-verifies only the un-verdicted items into `verify_<cell>_redo.workflow.js`; (c) running cells **sequentially, one Workflow at a time**. The redo runs came back with **0 failures** → 100 % coverage (NN 143 pass / 35 fail, SS 134 / 63).

**Subset-based fuzzy-dedup (new).** Cuisine dishes legitimately share ingredient/type tokens, so the old Jaccard-on-a-shared-≥5-char-token rule produced false merges that also *kept the generic over the specific* (e.g. dropped "Dal Bati Churma" for bare "Churma"; "Manapparai Murukku" for "Murukku"; merged Gongura Pachadi with Gongura Mamsam). `apply_verdicts.py` now merges only when one anchor's non-generic core is a strict **subset** of the other's, and keeps the **longer/more-specific** anchor. Result: every drop is a true variant (bare type-word → specific dish); distinct dishes sharing one ingredient survive.

**What Tier-1.5 caught (cuisine-specific):** spices/ingredients/crops not dishes (Alleppey green cardamom, Byadagi chilli, Attappady/Authoor GI crop varieties, Central Travancore jaggery), cuisine-**category** names ("Andhra cuisine", "Chettinad cuisine"), pan-Indian non-distinctive items (basundi, bhakri, boondi, chirote), city-name **leaks** (Alleppey, Bangalore), and beverages where a dish is required (Aval milk). Town-derived dish names correctly PASS fact_ok (Hyderabadi haleem, Bikaneri bhujia, Dharwad pedha) with leakage judged separately. The F3 language filter hard-rejected language-adjective leaks (Punjabi/Kashmiri X) before Claude — consistent with the soft-filter directive (only unambiguous leaks are deterministic; nuance goes to Claude).

**Cross-model (6/6, scored on one g6.12xlarge / 4×L4 via the fixed `launch_aws.sh`):** clean-RLHF preserves — Llama-3.1-8B corr **0.90 (N) / 0.89 (S)** (Δ +0.39 / +0.57); **Gemma-2-9B aligned sharper** (Δ −3.10 / −2.35, corr 0.92 / 0.90). **Mistral→Sarvam-M weakens binding — corr 0.44 / 0.49, Δ +3.62 / +3.51, 73–79 % base>aligned** — the Sarvam-M selectivity signal **continues into a third sub-concept** (Festivals → Costume +3.3–4.5 → Cuisine ~+3.5; directional, single confounded arm — Phase 4 adjudicates).

## A01-EE-03 + A01-WW-03 + A01-CC-03 — Cuisine row completed (wave 8)

Completes A01-03. Web-tier-dominated; every dish web-verified with a source URL via confirm-and-emit
research Workflows, then a Tier-1.5 verify Workflow per cell (run **sequentially**; all 100 % coverage on
the first run — no `--missing` redo). Released cells carry **0 state-name leaks, 0 bad counterfactuals
(r′≠r), 0 missing cross-validation, all 6 model_scores present**.

| Metric | A01-EE-03 | A01-WW-03 | A01-CC-03 |
|---|---|---|---|
| Candidates | 204 (SAN/Wiki 61 + web 143) | 165 (web 151 + stage2 14) | 47 (web 49 + MP sweep) |
| Stage-4 filtered | 199 | 165 | 47 |
| Tier-1.5 verified pass | 179 | 156 | 35 |
| After fuzzy-dup drop | 176 (−3) | 143 (−13) | 34 (−1) |
| Base-Llama gate (ΔL>1.0) | **170** (97 %) | **133** (93 %) | **30** (88 %) |
| Final | **100** | **100** | **30** (short) |
| States | 12 East (Bih 15 / Meg 13 / Sik 9 / Aru 9 / Od 8 / WB 7 / As 7 / Man 7 / Jhk 7 / Tri 6 / Nag 6 / Miz 6) | Guj 34 / Mah 34 / Goa 32 | MP 16 / Chh 14 |
| Token 1/2/3/4/5 | 15/47/22/7/9 | 100/0/0/0/0 | 0/0/16/0/14 |

**The headline result: West cuisine reaches a full 100, breaking the textile-ceiling assumption.**
A01-WW-02 (textile) was a 64-item short cell, so the handoff projected WW-03 short too. But cuisine is
**dish-dense** — Gujarat/Maharashtra/Goa yielded 143 verified distinctive dishes (a first sweep + a deep
long-tail sweep), comfortably clearing 100. So a region's cuisine ceiling tracks **dish density**, not the
count of eligible states. WW-03 is a clean **single-token** cell (all 3 West states are 1-token).

**Central is the genuine short cell (30), bottlenecked by leakage not sourcing.** MP's most famous dishes
are **city-name-bound** — Indori Poha/Sev, Bhopali Gosht/Korma — and correctly fail leakage_ok (the
capital/major-city giveaway rule), and several MP candidates are pan-Indian (Baati→Rajasthan, Khurma,
Lavang Lata→Bengal). A focused MP **long-tail + GI** ceiling-check sweep added 8 verified non-leaking MP
dishes (Kadaknath/Jhabua-GI, Tikkad, Bareta, Gajak/Morena, Chironji barfi/Sagar, the Malwa corn family
Bhutte ki Khichdi/Rabdi/Rassa), which both **confirmed the ceiling** and balanced the cell to 16/14. The
fuzzy-dedup correctly folded "Chakki ki Sabzi" into "Chakki ki Shaak" (sabzi is a generic type-word).

What Tier-1.5 caught (cuisine-specific, this wave): pan-Indian non-distinctive sweets (Balushahi,
Khurma), wrong-state mis-attributions (Bamboo Chicken→Araku Valley/AP not Arunachal; Bareta is a Punjab
town, not a dish), capital-city **leaks** (Indori/Bhopali), and generic descriptors ("Bamboo Shoot Curry").
Town-derived dish names correctly PASS fact_ok with leakage judged separately (Surti Locho, Kolhapuri
Mutton). counterfactual_ok ≈ 100 % (clean cuisine corruptor bank, cross-region length-matched).

**Cross-model (6/6, one fresh g6.12xlarge / 4×L4):** clean-RLHF preserves — Llama-3.1-8B corr **0.87 (E) /
0.92 (W) / 0.93 (C)** (Δ +0.94 / +1.07 / +0.40); **Gemma-2-9B aligned sharper** (Δ −2.74 / −1.75 / −3.12,
corr 0.84 / 0.94 / 0.94). **Mistral→Sarvam-M weakens binding — Δ +3.15 (E) / +2.35 (W) / +3.52 (C), corr
0.25 / 0.45 / 0.55, 53–77 % base>aligned.** The Sarvam-M cuisine selectivity signal **holds across the whole
row** (NN +3.62 / SS +3.51 / EE +3.15 / WW +2.35 / CC +3.52 ≈ +2.3–3.6); West is the mildest, plausibly
the clean single-token stratum (directional, single confounded arm — Phase 4 adjudicates).

## A01-NN-04 (North) + A01-SS-04 (South) — Rituals row Batch A, Tier-1.5 quality (released)

First Rituals cells (STEP-0 sub-concept-04 generalization). The Rituals construct is the **noisiest** in the
grid: SANSKRITI's Rituals_and_Ceremonies pool mixes true rites with festivals, dances/puppetry, modern
state/military ceremonies, and bare culture statements, so `_VERIFY_PROFILE["04"]` rejects festivals-proper,
secular dances, dishes/deities/temples/persons/crafts, military ceremonies (Beating Retreat), and genuinely
pan-Indian rites — while passing community/town/temple-named rites (Theyyam, Bhuta Kola, Tusu Puja) and
ritual-possession-theatre. Every rite is web-verified with a source URL.

| Metric | A01-NN-04 (North) | A01-SS-04 (South) |
|---|---|---|
| Candidates (web / stage2) | 148 / 1 | 143 / 5 |
| Stage-4 filtered | 147 | 146 |
| Tier-1.5 verified pass | 121 | 116 |
| Base-Llama gate (ΔL>1.0) | **108** (89 %) | **110** (95 %) |
| Final | **100** | **100** |
| Provenance gaps | 0 | 0 |
| States | 9 (Ukd 17 / Lad 16 / Pun 15 / Raj 14 / J&K 13 / UP 12 / HP 11 / Del 1 / Har 1) | 6 (Ker 28 / Kar 23 / TN 19 / AP 14 / Tel 13 / Pud 3) |
| Token 1/2/3/4 | 30 / 29 / 17 / 24 | 51 / 19 / 27 / 3 |

**What Tier-1.5 caught (ritual-specific):** festivals mislabelled as rites (Aadi Perukku, Dev Deepawali, Chhath,
Dosmoche/Galdan Namchot as gazetted festivals, Lavi Fair = trade fair), **secular dances / performing arts**
(Cham, Aati Kalenja, Terahtali, Ramnagar Ramlila ritual-theatre, Thoda martial archery), wrong-state
mis-attributions (Boddemma → Telangana not AP), genuinely pan-North multi-state folk-deity worship with no
single-state identity (Gugga/Gogaji across 6 states, Gauri Puja, Kuldevi Puja, Mata ki Chowki, Ganga Dussehra
snan), and a manufactured descriptor (Land scooping rite). counterfactual_ok ≈ 100 % (the expanded
cross-region ritual corruptor bank, length-matched 2–8 tokens).

**The Sikh-rite construct correction (the key quality finding).** The first North verify rejected distinctive
Sikh rites (Anand Karaj, Akhand Path) as "pan-Sikh therefore not single-state", which — combined with Punjab's
genuinely pan-North folk rites failing distinctiveness — collapsed Punjab to **4** and made NN-04 a 96-item
short cell. This was **over-strict**: *pan-Sikh is not pan-Indian.* Sikhism's homeland is Punjab (the only
Sikh-majority Indian state), so Sikh rites are Punjab's distinctive ritual heritage. The verify profile was
corrected (a rite of a religious/ethnic community whose homeland is the target state passes; only rites
practised across many states are pan-Indian — generalizes to Buddhist-monastic → Ladakh/Sikkim and tribal
rites), the two were restored, and a Sikh/Punjab top-up sourced (Nagar Kirtan, Dastar Bandi, Palki Sahib,
Sukhasan, Chaur Sahib seva, Prakash, Kar Seva, Bhog; folk Baba Sodal, Haider Sheikh, Jathera). Punjab went
4→15 and the cell 96→**100**. Decisively, the restored/added Sikh rites **passed the base-Llama ΔL gate** —
the model binds them to Punjab — so this is a construct-correct inclusion, not padding, and consistent with the
soft-filter directive (don't over-sanitize away real cultural signal). A reusable principle for the rest of the
row (and Axis A03 Religion).

**Cross-model (6/6) — Rituals breaks the uniform Sarvam-M weakening seen in Costume/Cuisine.** Clean-RLHF
preserves the binding (Llama-3.1-8B corr **0.94 N / 0.95 S**; Gemma-2-9B aligned **sharper**, Δ −1.54 / −1.61,
corr 0.90 / 0.94). The Indian-aligned **Mistral→Sarvam-M weakens it in the North (Δ +2.25, corr 0.64, 63/100
base>aligned — within the cuisine range) but NOT in the South (Δ +0.21, corr 0.65, balanced 29 base>aligned /
31 aligned>base — the weakest weakening of any cell built so far).** The most plausible reading: Sarvam-M's
Indian fine-tune *retains* South-Indian temple-ritual knowledge (Theyyam, Bhuta Kola, Kavadi, Tulu Daiva Kola)
that it down-weights for costume/cuisine — a regionally and sub-conceptually selective effect rather than a
flat across-the-board one (directional, single confounded arm — Phase 4 adjudicates).

## A01-EE-04 + A01-WW-04 + A01-CC-04 — Rituals row Batch B (released; completes Axis A01)

Same web-tier-dominated ritual chain; sourcing followed **SANSKRITI → Wikipedia depth-2 → web**. Every rite is
web-verified with a source URL. The base-Llama ΔL gate is **harsher for rituals than any prior sub-concept** (it
binds famous rites strongly but drops obscure tribal rites), so each cell took multiple confirm-and-emit research
rounds, verifying only new items (`--missing`) and resumable-rescoring only new item_ids.

| Metric | A01-EE-04 (East) | A01-WW-04 (West) | A01-CC-04 (Central) |
|---|---|---|---|
| Stage-4 filtered | 189 | 150 | 79 |
| Tier-1.5 verified pass | 156 | 122 | 61 |
| Base-Llama gate (ΔL>1.0) | **121** (78 %) | **103** (84 %) | **43** (70 %) |
| Final | **100** | **100** | **43** (short) |
| Provenance gaps | 0 | 0 | 0 |
| States | 12 East (WB/Od/As/Miz/Sik 11 … Jhk/Bih 5) | Maharashtra 55 / Gujarat 29 / Goa 16 | MP 25 / Chhattisgarh 18 |
| Token mix | 5/65/18/5/7 | 1-tok 100 | 3-tok 25 / 5-tok 18 |

**What Tier-1.5 caught (this wave):** festivals mislabelled as rites (Aoleang, Dol Jatra, Mera Houchongba,
Bhagoria, Bastar Lokotsav, various *melas*), **performing arts** (Bhaona, secular dance), pan-Indian rites
(Annaprashana, Angarki Sankashti, Annakut, Naam Kirtan, Maghi Purnima, Sanja), a **deity/place not a rite**
(Bhuria Baba, Kumdakot, and "rites" appended to bare temple names — Grishneshwar/Kapaleshwar/Mumbadevi), wrong-
state mis-attributions, and **descriptor-not-a-name** strings ("Dindi walking formation", "Shiv Vivah ceremony").
A recurring fix: the verifier twice doubted the curated corruptor **Nuala** (a real Punjab Gugga rite) — its
`counterfactual_ok` was restored from the corruptor bank (2 items).

**The gate, not sourcing, is the binding constraint for rituals.** EE was sourced to 189 filtered but gated to
121: the famous **Puri Jagannath cycle** (Snana Yatra, Chandan Yatra, Suna Besha, Pahandi, Chhera Pahanra,
Anasara) and **Bengali Durga rites** (Kumari/Sandhi/Sindur Khela) passed, while obscure NE tribal rites (Apokpa,
Bathow, Behdienkhlam) passed verify but dropped at the gate. **West reached 100** because Maharashtra is rite-
dense (Warkari Wari, jyotirlingas, Datta/Nath shrines, Devi cults, Ashtavinayak). **Central is short (43)**
because Chhattisgarh's Bastar/Gond rites are real but model-obscure (gate-fail); MP carries it via well-known
Ujjain/Narmada/temple rites. CC-04 (43) > CC-03 (30); released at real n.

**Cross-model (6/6):** clean-RLHF preserves (Llama corr **0.94 E / 0.90 W / 0.89 C**); **Gemma-2-9B aligned
sharper** (Δ −1.60 / −1.83 / −2.38). **Mistral→Sarvam-M weakens — Δ +1.69 (E) / +1.60 (W) / +1.40 (C)** (corr
0.64 / 0.63 / 0.54). The whole **Rituals row Sarvam-M signal — NN +2.25 / SS +0.21 / EE +1.69 / WW +1.60 /
CC +1.40 (mean ≈ +1.4) — is MILDER than Costume (+3.3–4.5) and Cuisine (+2.3–3.6), and South uniquely shows
none.** This is the strongest evidence yet that the Indian fine-tune's selectivity is **sub-concept- and
region-dependent** (it preserves South-Indian temple-ritual knowledge while down-weighting costume/cuisine).
Directional, single confounded arm — Phase 4 adjudicates. **Axis A01 (Regional Specificity) is COMPLETE: 20/20
cells, 1,778 items, all 6/6 model-scored.**
