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

## Known improvements for scaling to 60 cells
- **Corruptor quality (Stage 3):** the SANSKRITI corruptor extraction yields some malformed anchors;
  Claude caught them but it wastes candidates. Clean the corruptor pool / prefer Wikipedia corruptors.
- **F2 place-leak:** harden to hard-reject capital/major-city names in the anchor (Claude caught these).
- **Web tier:** authorized + designed but NOT needed here (108 ≥ 100). Resources would log to
  `data/resources/web_sources.jsonl` (url, domain, query, accessed_utc) if used.

## Source resources log (beyond Wikipedia/SANSKRITI)
General-internet sources for any top-up are recorded in `data/resources/web_sources.jsonl`. None used.
