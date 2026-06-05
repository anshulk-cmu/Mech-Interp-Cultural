# ICCD-6K Phase 1 — Pipeline (build doc)

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

Living doc, written stage-by-stage as code lands. Smoke target: cell **A01-SS-01**
(Regional Specificity / South / Festivals), 100 final items, then scale to 60 cells.

## Layout

```
iccd/                     # pipeline package
  config.py               # locked design: cells, regions, sub-concepts, seeds, paths, SANSKRITI map
  logutil.py              # console + per-stage file logging (logs/<stage>.log)
  state.py                # resumability: atomic write_json + JSONL checkpoints (resume on rerun)
  wiki.py                 # MediaWiki API client w/ on-disk cache + revision-id capture
  tok.py                  # pinned Llama-3.1 tokenizer (length match, token balance, delta-L)
  stages/
    stage0_bootstrap.py   # gazetteer + language_map (F2/F3 leakage resources)
    stage1_sanskriti.py   # ingest+filter SANSKRITI -> per-cell pool + cross-val index
    stage2_sourcing.py    # Wikipedia category walk -> ~240 candidates/cell
    stage3_minimalpairs.py# STR cross-anchor-swap clean/corrupt triplets (Axis A)
    stage4_filters.py     # F1-F7 deterministic filters
    stage5_sample.py      # token-balanced sampling (Pass A pre-gate / Pass B post-gate)
    stage6_provenance.py  # provenance audit + release assembly -> data/final/iccd_<cell>.json
    claude_verify.py      # Tier-1.5 Claude in-session verification (prepare/combine/apply)
scripts/
  scoring/  stage8_score.py (GPU scorer)  ec2_run*.sh + babel_suite.sbatch (runners)  teardown.ps1
  cross_model.py  combine_cells.py  (analysis / aggregation)
  dev/  inspect_sources.py
data/
  raw/        (gitignored)  SANSKRITI csv
  resources/  gazetteer.json language_map.json tokenizers.json
  interim/    (gitignored)  regenerable stage outputs
  final/      iccd_<cell>.json   (the release)
docs/  SETUP.md  PIPELINE.md  QUALITY.md  REPORT.md  CROSSMODEL.md  CELLS.md
```
Stage 8 model scoring is `scripts/scoring/stage8_score.py` (runs on the EC2), not a stages/ module.

## Resumability & logging (every stage)

- Whole-stage output via `state.write_json` (atomic temp+replace) — rerun skips if present.
- Per-item loops (Wikipedia fetch, Stage-8 scoring) append to a JSONL keyed by id; rerun
  reads done-keys and continues. Wikipedia responses cached on disk by query hash.
- Each stage logs to console and `logs/<stage>.log` with timestamps + per-step counts.

## Validated source facts (2026-05-31)

- SANSKRITI columns: `state, attribute, question, option1-4, answer, short explaination / source link, question_type`.
- 16 attributes; 4 question_types (`Association, Country Prediction, General Awareness, State Prediction`).
- SANSKRITI->cell map (clean 1:1 only): Festivals→A01-01, Costume→A01-02, Cuisine→A01-03,
  Rituals_and_Ceremonies→A01-04, Art→A02-03, Religion→A03-02, History→A03-03, Medicine→A03-04.
  - Known gaps (full build, not smoke): `Dance_and_Music` lumps B01+B02; no SANSKRITI attribute for
    B04 Architecture or C01 Caste → those B/C cells lean on Wikipedia/web (plan §5.1 limitation).

## Stage contracts

Filled per stage below as each is implemented and tested.

### Stage 1 — SANSKRITI ingest + filter
- **In:** `data/raw/sanskriti/Merged_Dataset_english_SANSKRITI.csv`
- **Ops:** keep qtype∈{Association, State Prediction}; map attribute→(axis,subconcept); normalize
  state→region; bucket to cells; build per-state text blob for cross-validation.
- **Out:** `data/interim/sanskriti_pool.json` (by cell), `data/interim/sanskriti_state_text.json`.
- **Note (smoke simplification):** SANSKRITI Limitation-4 answer-replacement has no flag column;
  the robust detector is deferred — Stage 1 drops only structurally broken rows and defers the
  uniquely-Indian judgment to cross-validation + Claude/human tiers. Documented, not silent.
- **Result:** 21,853 → 10,841 (qtype) → 5,816 bucketed; A01-SS-01 = 323 SANSKRITI items.

### Stage 0 — bootstrap (gazetteer, language_map, tokenizer pin)
- **Out:** `data/resources/{gazetteer.json, language_map.json, tokenizers.json}`.
- Gazetteer = district/city names per South state (F2). language_map hand-curated (F3).
  Tokenizer pinned: `meta-llama/Llama-3.1-8B` @ commit `d04e592b…`.

### Stage 2 — sourcing (inclusive, Claude does flexible cross-val)
- **In:** `sanskriti_pool.json`, `sanskriti_state_text.json`; live Wikipedia.
- **Ops:** anchors from SANSKRITI questions (3 regex forms) + Wikipedia festival categories.
  Keep a candidate if **SANSKRITI-attested (distinctive-token match) OR the Wikipedia intro
  confirms the festival→state fact**. Flags `sanskriti_attested`, `wiki_state_confirmed` recorded.
  We intentionally do NOT hard-reject on exact-string mismatch — flexible semantic cross-validation
  is Claude's job at Tier 1.5 (user directive), keeping valid items from being dropped on brittle matches.
- **Out:** `data/interim/candidates_raw_A01-SS-01.jsonl` (resumable). **Result:** 171 anchors → **126** kept.

### Stage 3 — minimal pairs (Axis A STR cross-anchor swap)
- **Methodology grounding:** ROME (s,r,o) tuple with a fixed relation template; Zhang & Nanda STR =
  swap the subject for a length-matched, in-distribution alternative, one detail differs
  (arXiv:2202.05262, 2309.16042; web-confirmed). Template: `"{festival} is a festival celebrated in
  the Indian state of"` → object = state.
- **Pair rules enforced:** corruptor is a same-sub-concept festival from a **different region**
  (so **r′ ≠ r**), token-length-matched anchor (relax ±1, logged), identical template suffix (only the
  anchor changes). Leakage (F2/F3) + corruption (F4) checks are Stage 4. Deterministic, seed 271.
- **Out:** `data/interim/pairs_A01-SS-01.json`. **Result (after broadening):** **172** pairs.
- e.g. clean `Urus … state of`→Andhra Pradesh vs corrupt `Kang … state of`→Manipur (r′).
- **Broadening to guarantee 100 final (user-approved):** Stage 2 made inclusive + recursive
  Wikipedia subcategory traversal (depth 1) + language-scoped festival categories → 237 anchors → 174 candidates.

### Stage 4 — F1-F7 filters (hard clear, flag borderline for Claude)
- **Hard reject:** F1 target>3 tokens, F5 suffix mismatch, F6/F7 prefix length (8-64), F4 corruption
  fail (r'==r or corrupted still contains anchor), F2 target STATE name in clean prefix (clear leak),
  F3 uniquely-state-identifying language (Tamil/Malayalam/Kannada).
- **Flag for Claude (not dropped):** gazetteer place inside the anchor; non-unique language (Telugu).
- **In:** `pairs_<cid>.json` (+ gazetteer/language_map). **Out:** `pairs_filtered_<cid>.json`, `rejected_<cid>.json`.
- **Result:** 172 → kept **165**, rejected 7 (6 F2_state_name_leak, 1 F3_unique_language_leak), 1 flagged.

### Stage 5 — token-balanced sampling (two passes around the model gate)
- **Pass A:** carry the 165 filtered pairs forward as the Stage-8 scoring batch
  `stage8_input_<cid>.json` (item_id, clean/corrupt prefix, target, r/r', target_tokens).
- **Pass B (after Stage 8 + Claude):** from survivors, draw final 100 with 50/30/20 one/two/three-token
  balance, **random within stratum (seed 42), never ranked on delta-L** (anti selection-on-DV).
  Realized token-length distribution recorded (Axis A targets are states → mix is state-constrained).

### Stage 8 — model-side delta-L gate (fp16 Llama-3.1-8B on AWS g6.xlarge)
- Runs on the EC2: load `meta-llama/Llama-3.1-8B` fp16, score each item's clean & corrupted prefix on
  the fixed target (leading-space, teacher-forced), delta-L = logP(clean) - logP(corrupted) nats.
  Retain |delta-L| > 1.0 (Axis A/B one-sided; Axis C two-sided). Logit/log-odds, never probability.
- **Web tier (user-approved):** if survivors < 100, top up via general-internet sourcing of more
  festival→state facts; every web resource (URL + access time + domain) logged for provenance.

### Stage 8 result (AWS g6.xlarge, fp16 Llama-3.1-8B)
- 165 scored, **163 kept** (delta-L > 1.0). Llama-3.1-8B knows these festival→state facts well
  (e.g. Bhogi→AP dL=9.3, Atla Tadde→AP dL=7.9; weak "Urus"→AP dL=0.39 correctly dropped).
- Instance/SG torn down (`scripts/scoring/teardown.ps1`). ~25 min uptime ≈ $0.35.

### Tier 1.5 — Claude verification (in-session subagents, 4 checks)
- 4 subagents over the 163 survivors → **108 approved / 55 flagged**. Caught (flexibly, as intended):
  non-festivals the recursive Wikipedia walk pulled in (temples/monastery/sports/shop → fact_ok),
  place-name leakage F2 only flagged (Thrissur/Hyderabad/Chennai/Mysore/Bengaluru → leakage_ok),
  malformed corruptors from SANSKRITI corruptor extraction (mojibake, "Ganesh Chaturthi in maharashtra",
  "Temple fairs (Jatras)" → natural_ok). The 108 that passed all carry clean corruptors.
- `combine()` → `claude_verdicts_<cid>.json`; `apply()` → `stage8_survivors_<cid>.json` (108 ids).

### Stage 5 Pass B + Stage 6 — final 100
- Pass B drew **100 final** (random within token-length stratum, seed 42). Realized 1/2/3-token mix
  **61/17/22** (target 50/30/20; 2-token capped — Tamil Nadu is the only 2-token South state, recorded).
- Stage 6 release assembly: **100 items, 0 missing cross-validation** (release gate). Each record carries
  delta-L, Claude verdict, wiki oldid, cross_validated_by → `data/final/iccd_A01-SS-01.json`.

## Funnel (A01-SS-01)
```
SANSKRITI cell items 323 -> Stage2 candidates 174 -> Stage3 pairs 172 -> Stage4 kept 165
 -> Stage8 model-kept 163 -> Claude-approved 108 -> Pass B final 100 (0 cross-val gaps)
```

## Scale-out: A01-NN-01 (North) + A01-EE-01 (East) — built through Tier-1.5, pre-Stage-8

Second build wave. Same nine-stage chain as A01-SS-01, driven per cell by `scripts/build_cell.py`
(reusable: `bootstrap`, `stage2`, `webtier`, `pairs`, `all`). Claude Tier-1.5 was run *before*
the Stage-8 gate (the gate needs AWS/Babel), so each cell's `stage8_input_<cell>.json` is the
over-provisioned **verified** batch, ready to score.

### Funnels
```
A01-NN-01: candidates 208 (SANSKRITI 7 / Wiki 63 / web 138)
 -> Stage3 pairs 197 -> Stage4 F1-F7 kept 155 -> Tier-1.5 Claude pass 126 (deduped)
 -> Stage8 base-Llama-3.1-8B ΔL>1.0 gate 119 -> Pass B 100 (token 50/30/20)

A01-EE-01: candidates 214 (SANSKRITI 20 / Wiki 136 / web 58)
 -> Stage3 pairs 209 -> Stage4 F1-F7 kept 168 -> Tier-1.5 Claude pass 121 (deduped)
 -> Stage8 base-Llama-3.1-8B ΔL>1.0 gate 112 -> Pass B 100 (token 7/77/16)
```
Stage-4 rejections were dominated by **F1 token-overlong**: the Axis-A target IS the state, and
states whose names tokenize to >3 (Jammu and Kashmir, Himachal Pradesh, Chandigarh in North;
Arunachal Pradesh, Jharkhand in East) are excluded by design. North therefore draws from 7
short-name states, East from 10. Verified pools: 0 provenance gaps, 0 state-name leaks, 0
bad counterfactuals; provenance NN 90 Wikipedia-oldid + 36 web-URL, EE 112 + 9.

### Resources extended for the scale-out (all 60-cell-reusable)
- `data/resources/gazetteer.json`, `language_map.json` — Stage-0 walk extended South→all 30
  South/North/East states (district/city names for F2; language map for F3).
- `iccd/stages/stage4_filters.py` `_UNIQUE_LANG` — North/East single-state languages added
  (Punjabi, Odia, Assamese, Meitei, Khasi, Mizo, Kokborok, Maithili, …); shared languages
  (Hindi, Bengali, Santali, Nepali) stay flag-only.
- `iccd/stages/stage2_sourcing.py` `_EXTRA_CATS` — added the real language-community Wikipedia
  festival categories that exist (`Punjabi festivals`, `Bengali festivals`, `Bengali Hindu festivals`).
- `data/resources/corruptor_bank.json` — added the South (`SS`) corruptor list so North/East
  cells have a rich cross-region swap pool (now all 5 regions present).
- `data/resources/web_festivals.json` — `A01-NN-01` (138) + `A01-EE-01` (58) curated web-tier
  festivals, every entry web-verified with a source URL (provenance logged to `web_sources.jsonl`).

### Hardening landed this wave (helps every remaining cell)
- `iccd/wiki.py`: retry/backoff on transient drops (RemoteDisconnected/timeout) over the on-disk cache.
- `iccd/stages/stage2_sourcing.py`: U+FFFD mojibake guard + Latin-diacritic ASCII folding in `_clean`
  (e.g. "Chapchâr Kût"→"Chapchar Kut"), so Northeast titles become clean, deduped, natural anchors.
- `iccd/stages/stage2b_webtier.py`: web entries may carry a `wikipedia_title` hint for a reliable oldid.

### Tooling added
- `scripts/build_cell.py` — local per-cell driver (Stages 2→5 Pass A) + `bootstrap` for new regions.
- `scripts/gen_verify_workflow.py` — emits a batched Tier-1.5 verification Workflow for a filtered cell.
- `scripts/finalize_cell.py` — post-Stage-8: (Claude-pass ∩ ΔL>1.0) → Pass B 100 → Stage 6 release.

### How these two cells were finished (Stage 8, done)
Scored the 6-model suite on the combined `suite_input_NNEE.json` (247 verified items), **in parallel**:
4 small models (Llama-3.1-8B base/Instruct, Gemma-2-9b base/it) on an **AWS g6.xlarge** (L4 24GB) via
`ec2_run_suite_nnee.sh`; the 2×24B (Mistral-Small-24B-Base, Sarvam-M) on **CMU Babel** (A6000×4) via
`babel_suite_nnee.sbatch` (job 8227040). Base-Llama scores (threshold −1000) were re-thresholded at
ΔL>1.0 per cell to form the gate → `scripts/finalize_cell.py <cell>` → `data/final/iccd_<cell>.json`
→ `scripts/cross_model.py <cell>` appended all 6 models' `model_scores` + `cross_model_delta` →
`scripts/combine_cells.py` (manifest 3/60, 300 items). AWS instance + SG torn down; Babel token/dir/cache wiped.

## Scale-out wave 3: A01-WW-01 (West) + A01-CC-01 (Central) — released, 6/6 models

The two festival-thin regions (SANSKRITI West 132 / Central 104). Same chain, but the
candidate pool came overwhelmingly from the **web tier** (Wikipedia/SANSKRITI yield only
~35 / ~15), so this wave exercised the designed-but-unused-until-now web path at full scale.

### Funnels
```
A01-WW-01: candidates 164 (SANSKRITI 5 / Wiki 30 / web 129)
 -> Stage3 pairs 147 -> Stage4 F1-F7 143 -> Tier-1.5 Claude 121 -> 111 (−10 fuzzy-dup)
 -> Stage8 base-Llama ΔL>1.0 gate 105 -> Pass B 100  (Maharashtra 40 / Gujarat 30 / Goa 30; token 100/0/0)

A01-CC-01: candidates 143 (SANSKRITI 1 / Wiki 14 / web 128)
 -> Stage3 pairs 134 -> Stage4 F1-F7 134 -> Tier-1.5 Claude 118 -> 113 (−5 fuzzy-dup)
 -> Stage8 base-Llama ΔL>1.0 gate 102 -> Pass B 100  (Madhya Pradesh 58 / Chhattisgarh 42; token 58/42 over 3/5-tok)
```
Provenance: WW 72 Wikipedia-oldid + 28 web-URL; CC 54 + 46; both 0 gaps. West realizes a
clean **single-token** stratum (Gujarat/Maharashtra/Goa all 1 tok; DNH&DD dropped at 13 tok).

### F1 design exception (Central / Chhattisgarh)
Central has only two states under the F1 ≤3-token target cap candidate set: Madhya Pradesh
(3 tok) and Chhattisgarh — which the Llama-3.1 tokenizer splits into **5** tokens. Dropping it
(as J&K/Himachal/Arunachal/Jharkhand are dropped elsewhere) would collapse Central to a
single state and make 100 distinctive items infeasible. So Chhattisgarh is kept as a
**documented exception** in `config.F1_TOKEN_EXCEPTIONS` (consumed by `stage4_filters`); its
items form a 5-token stratum carried through Pass B's fallback fill. The cap's purpose
(token-length confound control) is preserved by documenting the stratum, and the ΔL gate +
Tier-1.5 verification are length-agnostic. User-approved 2026-05-31.

### Web tier at scale (Workflow-driven)
Sourcing was a **structured-output web-research Workflow**: one general-purpose agent per
state×scope (folk/tribal/temple-jatra and mela/community), each web-verifying every festival
with a logged source URL and emitting `distinctive=true` only. West ran one pass (5 states ×
2 scopes); Central, thinner, took a second supplemental pass (MP×2 + Chhattisgarh×2, deeper
regional/tribal scopes) to clear the ≥130-filtered over-provision bar. A Tier-1.5 verify
Workflow then fact-checked each minimal pair; fuzzy-dup variants (Zatra/Jatra, Madhavpur
Ghed/Fair, Akti/Akti Tihar) were dropped before the gate.

### Tooling added this wave (all 60-cell-reusable)
- `scripts/merge_web_festivals.py` — append-safe merge of a web-research Workflow's output
  into `web_festivals.json` (ASCII-fold, distinctive-only, dedup vs existing + candidates_raw).
- `scripts/apply_verdicts.py` — save Tier-1.5 verdicts, fuzzy-dedup variant festival names,
  regenerate `stage8_input_<cell>.json` = the deduped Claude-pass batch.
- `scripts/rethreshold.py` — split the base-Llama suite scores per cell at ΔL>1.0 → `stage8_results_<cell>.jsonl`.
- `scripts/scoring/{ec2_run_suite_wwcc.sh, babel_suite_wwcc.sbatch}` — WWCC suite runners.
- Resources extended: `gazetteer.json` + `language_map.json` (Stage-0 walk, now all 36
  South/North/East/West/Central states), `stage4._UNIQUE_LANG` (Gujarati/Marathi/Konkani/
  Chhattisgarhi), `stage2._EXTRA_CATS` (Marathi/Gujarati/Goan festival categories),
  `web_festivals.json` (`A01-WW-01` 129, `A01-CC-01` 128, every entry web-verified).

### Stage 8 (done, parallel AWS + Babel)
Combined `suite_input_WWCC.json` (224 verified items). 4 small models (Llama-3.1-8B base/it,
Gemma-2-9b base/it) on an **AWS g6.xlarge** (L4 24GB; us-east-1a after a/b/c/d/e/f capacity
hunt) via `ec2_run_suite_wwcc.sh`; the 2×24B (Mistral-Small-24B-Base, Sarvam-M) on **CMU
Babel** (A6000×4, job 8227432) via `babel_suite_wwcc.sbatch`. Re-threshold → `finalize_cell`
→ `cross_model` → `combine_cells` (manifest **5/60, 500 items**). AWS instance terminated + SG
deleted; Babel token/dir/scratch-cache wiped.

## Sub-concept generalization (STEP 0): A01-02 Costume & Textile (first non-Festivals build)

Waves 1-3 were all **A01-01 Festivals**, so several stages were festival-hardcoded. Before
building the first non-Festivals cells (A01-NN-02 North, A01-SS-02 South — Costume & Textile)
the pipeline was **parametrized by sub-concept** so 03 Cuisine and 04 Rituals reuse it unchanged.
A01-01 behavior is bit-for-bit preserved (the released festival cells are unaffected).

- **STR relation frame (Stage 3) is per sub-concept.** `config.RELATION_TEMPLATES[(axis, sub)]`
  holds the one fixed STR frame per cell (only the anchor token varies within a pair, so the
  suffix stays identical — strict STR). A01-01 = `"{anchor} is a festival celebrated in the Indian
  state of"` (verbatim); **A01-02 = `"{anchor} is a folk textile tradition associated with the
  Indian state of"`** (plan §D.2). `stage3_minimalpairs.py` reads it via `relation_template(axis, sub)`.
  03/04 carry provisional defaults, finalized when built.
- **Corruptors are sub-concept-matched (Stage 3).** `corruptor_bank.json` is restructured to
  `by_subconcept[sub][region]`; `_corruptor_pool(region, sub)` selects the matching pool (legacy
  flat region keys still read as "01"). A reusing-festival-corruptors bug ("Pongal is a folk textile
  tradition…") is structurally impossible now. Added a clean, single-state-distinctive **Costume
  set ("02")** across all 5 regions (Phulkari/Punjab, Kanjeevaram/TN, Pochampally ikat/Telangana,
  Sambalpuri/Odisha, Muga silk/Assam, Patola/Gujarat, Paithani/Maharashtra, Chanderi/MP, Kosa
  silk/Chhattisgarh, …; ~26-item cross-region pool per cell, token-length 2-6 for length matching).
- **Wikipedia sourcing is per sub-concept (Stage 2).** `_SUBCONCEPT_SOURCING[(axis, sub)]` gives
  `state_cats` / `extra_cats` / `broad_cats`. Festival cats are per-state. For textiles the
  per-state cats (`Textile arts of {s}`, `Crafts of {s}`) are **empty on en.wikipedia** (probed
  2026-06-01), but India-level **broad cats** are rich — `Textile arts of India` (183), `Saris`
  (61), `Embroidery in India` (35), `Indian clothing`. Broad cats are fetched **once** and each
  member's state is **resolved from its Wikipedia extract restricted to the cell's region** (kept
  only if exactly one region-state is named), so a shared category can't misattribute. `_GENERIC`
  (the SANSKRITI-attestation skip set) gained costume vocabulary (saree/textile/weave/embroidery/…).
- **Tier-1.5 verify prompt is per sub-concept.** `gen_verify_workflow.py` injects a
  `_VERIFY_PROFILE[sub]` (binding sentence + `fact_ok` definition). A01-02 `fact_ok` requires a
  *real, distinctive regional textile/weave/handloom/garment/embroidery of the target state*
  (GI-tag / handloom-board aware), rejecting festivals/food/dance/people/places and pan-Indian
  non-distinctive garments; leakage_ok notes that toponymic craft names (Banarasi, Sambalpuri,
  Chanderi) are the normal convention and pass unless the explicit STATE name is present.
- **Release record is per sub-concept (Stage 6).** `stage6_provenance.py` no longer hardcodes
  `sub_concept:"Festivals"`/`axis:"A01"`; it reads `SUBCONCEPTS[axis][sub]` and the cell's axis.
- **Costume leakage nuance.** Many textiles are toponymic (Banarasi←Banaras/UP, Sambalpuri←Sambalpur/
  Odisha, Pochampally←a Telangana town). The verbatim-state-name F2 rule + gazetteer city flag +
  Tier-1.5 adjudication handle this; the explicit-city variant ("Kanchipuram silk") fails leakage
  while the adjective form ("Kanjeevaram") passes — the desired selection.

## Costume wave: A01-NN-02 (North) + A01-SS-02 (South) — released, 6/6 models

First **non-Festivals** cells, built on the STEP-0 generalization above.

### Funnels
```
A01-NN-02: web-tier (SANSKRITI/Wiki costume-thin; Wikipedia broad-cats + 4 web passes)
 -> Stage3 pairs 156 -> Stage4 F1-F7 146 -> Tier-1.5 Claude 123 -> 110 (-13 fuzzy-dup)
 -> Stage8 base-Llama ΔL>1.0 gate 101 -> Pass B 100  (UP 28 / Rajasthan 26 / Punjab 21 /
    Uttarakhand 15 / Ladakh 8 / Haryana 2; token 47/38/15)

A01-SS-02: same chain + a focused top-up pass
 -> Stage3 pairs 142 -> Stage4 F1-F7 142 -> Tier-1.5 Claude 127 -> 107 (-20 fuzzy-dup)
 -> Stage8 gate 106 / survivors 104 -> Pass B 100  (Karnataka 25 / TN 24 / AP 19 /
    Kerala 16 / Telangana 16; token 41/24/35)
```
**F1 exclusions:** North drops Himachal Pradesh + Jammu & Kashmir (the two textile-richest North
states, >3 tokens) — a real coverage cost, documented; Delhi/Chandigarh have no distinctive regional
textile tradition. South drops nothing (all 5 eligible states are textile-rich). Provenance: every
web textile carries a source URL (GI registry / Ministry of Textiles / Wikipedia / craft archives);
0 cross-validation gaps on either cell.

### What was different from Festivals
- **SANSKRITI is costume-thin in practice:** its costume questions don't match the festival-shaped
  anchor regex, yielding only ~6–8 clean anchors/cell, so both cells are **web-tier-dominated** even
  for "rich" South — heavier than the SANSKRITI inventory suggested. Took 4 (North) / 5 (South)
  Workflow web passes to over-provision ≥130 filtered (declining yield: 180 → 77 → 31 → 27 distinct).
- **ΔL gate retention ≈ 92–96 %** — as strong as Festivals. Llama-3.1-8B binds well-documented GI
  textiles strongly; obscure revival/village weaves drop, but Tier-1.5 already filters to documented
  GI textiles, so retention stays high. (Earlier worry that textiles would gate poorly was unfounded.)
- **Fuzzy-dedup is the big textile attrition:** South has many `saree`/`sari`/`silk` name-variants of
  one textile; `apply_verdicts` (now textile-suffix-aware) dropped 13–20/cell.
- **Two scoring rounds for South:** SS-02 first landed 97 after the gate; a small top-up (9 newly
  sourced famous GI textiles — Madras checks, Salem Fabric, Khun, Gajendragad, Kuthampully/
  Chendamangalam dhoti, Kanchi cotton, Elampillai) was sourced → verified → scored in a **resumable
  second GPU round** (results appended to the six `results_<slug>.jsonl`) → re-threshold → 100.

### Stage 8 (done, parallel AWS + Babel)
Combined `suite_input_NN02SS02.json` (211 items round 1, +9 round 2). 4 small models on an **AWS
g6.xlarge** (L4 24 GB) via `ec2_run_suite_nn02ss02.sh`; the 2×24B (Mistral-Small-24B-Base, Sarvam-M)
on **CMU Babel** (A6000×4) via `babel_suite_nn02ss02.sbatch`. **Babel hardening this wave:** Babel's
base miniconda had been upgraded to huggingface_hub 1.x (httpx backend) + transformers 5.x, whose
multi-shard 24B downloader fails with "Cannot send a request, as the client has been closed"; the
sbatch now pins `transformers==4.48.3` + `huggingface_hub<1.0` to `--user` (keeps base torch,
non-destructive). Re-threshold → `finalize_cell` → `cross_model` → `combine_cells` (manifest **7/60,
700 items**). Both AWS instances + SGs torn down; Babel token/dir/scratch-cache wiped. *(Future waves:
the AWS G/VT vCPU quota is now 48, so the 24B run can move to a second AWS box — a g6.12xlarge /
g6e.12xlarge 4-GPU instance — instead of Babel; see the EE-02/WW-02 handoff.)*

## Costume wave 2: A01-EE-02 (East) released; A01-WW-02 (West) PAUSED at a structural ceiling

Built A01-EE-02 to release and discovered a **design-level limit** for the few-eligible-state Costume cells.

### A01-EE-02 (East) funnel
```
candidates: Wikipedia broad-cat (48) + 3 web-research passes (20-agent sweep, 12-agent retry,
            5-agent rich-state top-up; every textile web-verified + source URL)
 -> Stage3 pairs 163 -> Stage4 F1-F7 156 -> Tier-1.5 Claude 131 -> 113 (-18 fuzzy-dup)
 -> Stage8 base-Llama ΔL>1.0 gate 108 (96% retention) -> Pass B 100
    (Odisha 16 / Assam 16 / WB 14 / Manipur 12 / Mizoram 9 / Meghalaya 9 / Bihar 8 /
     Nagaland 7 / Tripura 6 / Sikkim 3; token 8/80/12). Jharkhand (4 tok) & Arunachal (5 tok)
     F1-excluded. 0 missing cross-validation.
```
Cross-model: Llama corr 0.92 (Δ +0.85), Gemma aligned sharper (corr 0.92, Δ −2.31), **Mistral→Sarvam-M
corr 0.47, Δ +4.28 (88/100 base>aligned) — the strongest costume weakening of any cell** (NN +3.32 → SS +3.91 → EE +4.28).

### Verify/research Workflow hardening (landed in `scripts/gen_verify_workflow.py`; reusable)
Schema-forced `general-purpose` subagents that web-search heavily **exhaust their turn budget and never
call StructuredOutput** (observed 0/14, 1/10, 8/20 emit rates). Fixes:
- **Verify:** judge from the provided evidence + own knowledge; web-search only the doubtful few
  (budget ≤5); a `CRITICAL OUTPUT RULE` makes the StructuredOutput call mandatory; **batch size 7**.
  After this, emit rate = 100 % (20/20, 23/23, 13/13).
- **Research:** **confirm-and-emit** — give each agent an EXPLICIT candidate list (known GI/handloom
  inventories) so it confirms rather than open-ended-searches, cap searches, force the emit. Open-ended
  "dig into the long tail" prompts fail; explicit-seed "confirm these, EMIT" prompts succeed.

### Stage 8 both-tiers-on-AWS (no Babel) + gotchas
The G/VT vCPU quota is 48, so all 6 models scored on AWS. **Small (g6.xlarge=4 vCPU) + large
(g6.12xlarge=48) cannot run concurrently (52 > 48)** — run sequentially, or consolidate onto the large
box. Gotchas learned (now in `stage8_score.py` + the CC-02 handoff): **pull results BEFORE stopping** (a
stopped g6.xlarge may not restart — no AZ capacity); **Gemma-2 crashes under multi-GPU `device_map='auto'`**
(sliding-window cache device mismatch) → pin the ≤9B Gemma to a single GPU (`device_map={"":0}`, now the
default in `stage8_score.py`); **don't trust only the end-of-run marker** — a mid-suite `set -e` crash leaves
the box idle. `scripts/scoring/ec2_recover_ee02.sh` is the per-model-independent recovery runner.

### A01-WW-02 (West) + A01-CC-02 (Central) — released as short cells (wave 6); deep-search recalibration + single-box scoring
Axis-A targets ARE the state, so the F1 ≤3-token cap + frozen §3.2 taxonomy fix per-region eligibility: North 6 /
South 5 / East 10 — but **West 3** (Gujarat/Maharashtra/Goa) and **Central 2** (MP + Chhattisgarh). Textiles per
state are finite, so 100 is structurally unreachable. Before releasing short, two independent exhaustive sweeps
**converged** (confirming the ceiling is real, not under-sourcing): (a) an LLM web sweep (15 agents, GI/handloom
long-tail) and (b) a deterministic **deep-Wikipedia category crawl** — new tool `scripts/dev/wiki_deep_ceiling.py`
(crawls a wide textile category tree at depth-3, resolves the state from a ~2k-char extract, dedups vs the held
set) — which added ZERO valid textiles (only people/firms/castes/wrong-state). Also verified a ≤5-token cap adds
no states (West's only over-cap state = 13 tok; Chhattisgarh 5-tok already admitted). Released at real n via
`finalize_cell.py` (warns <100):
- **A01-WW-02 (West):** 3 sweeps (57 held + 23 new long-tail) → 114 pairs → 87 Tier-1.5 → 77 (−10 dup) → **64 gate (83 %)** → **64 final** (Guj 43 / Mah 18 / Goa 3; clean 1-tok stratum).
- **A01-CC-02 (Central):** Wiki/SANSKRITI 9 + 3 sweeps → 53 pairs → 46 Tier-1.5 → 43 (−3 dup) → **41 gate (95 %)** → **41 final** (MP 22 / Chh 19; 3-tok + 5-tok Chhattisgarh exception).

**Stage 8 — standard setup LOCKED (user, 2026-06-01): ONE g6.12xlarge (4×L4), all 6 models consolidated** — ≤9B on
`CUDA_VISIBLE_DEVICES=0`, the two 24B sharded over 4 GPUs; runner `scripts/scoring/ec2_run_suite_wwcc02.sh`. A
stopped box often can't restart (capacity) → fresh `launch_aws.sh` (AZ-hop; `SPOT=1` for spot, quota 48). A
**models-preloaded AMI** (all 6 cached on a 500 GB root EBS at `/home/ubuntu/hfcache`, built by
`scripts/scoring/bake_models.sh`) removes the ~25-min per-wave re-download. Cross-model extends the Sarvam-M
costume signal: Δ +4.48 (W) / +4.18 (C), corr 0.19 / 0.43.

## Cuisine row (03): A01-NN-03 + A01-SS-03 (wave 7) — and the permanent ≤5-token cap

Reused the STEP-0 sub-concept generalization, adding the cuisine block: `RELATION_TEMPLATE ("A01","03")`, `_SUBCONCEPT_SOURCING` cuisine cats (per-state `Cuisine of {s}`/`{s} cuisine` + community cats + India-level dish broad-cats), `gen_verify` `"03"` dish profile, cuisine `_GENERIC` sets, and a `corruptor_bank` `"03"` section. A01-01/02 build bit-for-bit unchanged. Both regions are **data-rich → both reached the full 100**.

**Permanent token-cap change (user-authorized 2026-06-05):** `MAX_TARGET_TOKENS` 3→5, `TOKEN_BALANCE={1:30,2:25,3:20,4:15,5:10}`, `F1_TOKEN_EXCEPTIONS` emptied. Eligibility expands (NN +Himachal Pradesh/J&K, SS +Puducherry/Lakshadweep, EE +Jharkhand/Arunachal). Cells built before this date used ≤3; **Cuisine (03) onward uses ≤5** (plan §13 v1.4).
- **A01-NN-03 (North):** 187 cand (SAN 13 / Wiki 47 / web 127) → 186 pairs → 178 F1-F7 → 143 Tier-1.5 → 139 (−4 dup) → **124 gate (89 %)** → **100 final** (9 states incl. J&K 11 / Himachal 10; token 37/31/11/21).
- **A01-SS-03 (South):** 207 cand (SAN 12 / Wiki 154 / web 41) → 204 pairs → 197 F1-F7 → 134 Tier-1.5 → 129 (−5 dup) → **128 gate (99 %)** → **100 final** (6 states incl. Puducherry 6; token 42/20/32/6). Wiki yield 154 = richest of any cell.

**Verify/dedup hardening:** schema-forced verify agents hit turn-budget emit-failures at high concurrency (two cells' Workflows at once → ~25 % emit); fixed by a hardened emit rule (judge from knowledge, ≤2 searches) + `gen_verify_workflow.py --missing` redo mode + running cells **sequentially**. Cuisine fuzzy-dedup (`apply_verdicts.py`) rewritten to **subset-based** — a bare type-word ("Churma"/"Murukku"/"Appam") folds into the specific dish ("Dal Bati Churma"…), keeping the distinctive name; the old Jaccard-on-shared-token merged different dishes sharing one ingredient (Gongura Pachadi vs Gongura Mamsam).

**Infra fixes (this wave):** `launch_aws.sh` root volume 200→500 GB — the AMI snapshot is 500 GB, so the old 200 made every launch fail with a masked `InvalidBlockDeviceMapping` reported as "InsufficientInstanceCapacity" (stderr piped to /dev/null; use `--dry-run` to surface real errors). A live gp3 throughput bump **125→1000 MB/s** cured cold-from-snapshot EBS slow loads (Gemma-base cold ~25 min → post-bump models ~2–3 min). **No HF token needed** (cached models load from the baked AMI). New **parallel standard runner** `scripts/scoring/ec2_run_suite_nnss03.sh`: four ≤9B models in parallel one-per-GPU (`CUDA_VISIBLE_DEVICES=0/1/2/3`, `wait`), then two 24B sequential sharded, no cache deletion. Cross-model: **Sarvam-M weakening continues — Δ +3.62 (N) / +3.51 (S), corr 0.44 / 0.49** (Festivals→Costume +3.3–4.5→Cuisine ~+3.5). Box stopped, not terminated.

## Known improvements for scaling to 60 cells
- **Corruptor quality (Stage 3):** the SANSKRITI corruptor extraction yields some malformed anchors;
  Claude caught them but it wastes candidates. Clean the corruptor pool / prefer Wikipedia corruptors.
- **F2 place-leak:** harden to hard-reject capital/major-city names in the anchor (Claude caught these).
- **Web tier:** authorized + designed but NOT needed here (108 ≥ 100). Resources would log to
  `data/resources/web_sources.jsonl` (url, domain, query, accessed_utc) if used.

## Source resources log (beyond Wikipedia/SANSKRITI)
General-internet sources for any top-up are recorded in `data/resources/web_sources.jsonl`. None used.
