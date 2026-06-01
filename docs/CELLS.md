# ICCD-6K — 60-Cell Status Tracker

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

60 cells = **3 axes × 5 regions × 4 sub-concepts**, 100 items each (6,000 total).
Cell id = `A0{axis}-{REGION}-0{sub}`, e.g. `A01-SS-01`. Regions: NN North · SS South · EE East · WW West · CC Central.

**Legend:** ✅ done · 🟡 in progress · ⬜ not started.
**Per-cell stages:** source → STR pairs → F1-F7 filters → base-model gate → Claude verify → final 100 → 6-model cross-model scoring.

**Summary: 5 / 60 released (A01-SS-01, A01-NN-01, A01-EE-01, A01-WW-01, A01-CC-01), 500 items, all 6/6 models scored.**
SANSKRITI festival inventory by region (from Stage 1) noted where known, to gauge sourcing ease.

---

## Axis A01 — Regional Specificity  (target answer = the state)

| Sub-concept | NN North | SS South | EE East | WW West | CC Central |
|---|---|---|---|---|---|
| 01 Festivals | ✅ A01-NN-01 | ✅ A01-SS-01 | ✅ A01-EE-01 | ✅ A01-WW-01 | ✅ A01-CC-01 |
| 02 Costume & Textile | ⬜ A01-NN-02 | ⬜ A01-SS-02 | ⬜ A01-EE-02 | ⬜ A01-WW-02 | ⬜ A01-CC-02 |
| 03 Cuisine | ⬜ A01-NN-03 | ⬜ A01-SS-03 | ⬜ A01-EE-03 | ⬜ A01-WW-03 | ⬜ A01-CC-03 |
| 04 Rituals & Ceremonies | ⬜ A01-NN-04 | ⬜ A01-SS-04 | ⬜ A01-EE-04 | ⬜ A01-WW-04 | ⬜ A01-CC-04 |

## Axis A02 — Cultural Flattening  (target = the indigenous term; description-based counterfactual)

| Sub-concept | NN North | SS South | EE East | WW West | CC Central |
|---|---|---|---|---|---|
| 01 Classical Dance | ⬜ A02-NN-01 | ⬜ A02-SS-01 | ⬜ A02-EE-01 | ⬜ A02-WW-01 | ⬜ A02-CC-01 |
| 02 Classical Music | ⬜ A02-NN-02 | ⬜ A02-SS-02 | ⬜ A02-EE-02 | ⬜ A02-WW-02 | ⬜ A02-CC-02 |
| 03 Visual Art | ⬜ A02-NN-03 | ⬜ A02-SS-03 | ⬜ A02-EE-03 | ⬜ A02-WW-03 | ⬜ A02-CC-03 |
| 04 Architecture & Built Form | ⬜ A02-NN-04 | ⬜ A02-SS-04 | ⬜ A02-EE-04 | ⬜ A02-WW-04 | ⬜ A02-CC-04 |

## Axis A03 — Sensitive Policy  (target = the topic name; description-based, effect direction REVERSED)

| Sub-concept | NN North | SS South | EE East | WW West | CC Central |
|---|---|---|---|---|---|
| 01 Social Structure & Caste | ⬜ A03-NN-01 | ⬜ A03-SS-01 | ⬜ A03-EE-01 | ⬜ A03-WW-01 | ⬜ A03-CC-01 |
| 02 Religion & Scripture | ⬜ A03-NN-02 | ⬜ A03-SS-02 | ⬜ A03-EE-02 | ⬜ A03-WW-02 | ⬜ A03-CC-02 |
| 03 History & Political Memory | ⬜ A03-NN-03 | ⬜ A03-SS-03 | ⬜ A03-EE-03 | ⬜ A03-WW-03 | ⬜ A03-CC-03 |
| 04 Traditional Medicine | ⬜ A03-NN-04 | ⬜ A03-SS-04 | ⬜ A03-EE-04 | ⬜ A03-WW-04 | ⬜ A03-CC-04 |

---

## Detailed status (cells with any progress)

| Cell | source | pairs | filters | gate | Claude | final n | cross-model |
|---|---|---|---|---|---|---|---|
| **A01-SS-01** | ✅ 144 (SANSKRITI+Wiki+web) | ✅ 151 | ✅ 144 | ✅ 139 (Llama-3.1-8B) | ✅ 106 pass | ✅ **100** | ✅ **6/6 models** (Llama base/it, Gemma base/it, Mistral-24B/Sarvam-M) |
| **A01-NN-01** | ✅ 208 (SANSKRITI 7 / Wiki 63 / web 138) | ✅ 197 | ✅ 155 | ✅ 119 (Llama-3.1-8B) | ✅ 126 pass | ✅ **100** | ✅ **6/6 models** |
| **A01-EE-01** | ✅ 214 (SANSKRITI 20 / Wiki 136 / web 58) | ✅ 209 | ✅ 168 | ✅ 112 (Llama-3.1-8B) | ✅ 121 pass | ✅ **100** | ✅ **6/6 models** |
| **A01-WW-01** | ✅ 164 (SANSKRITI 5 / Wiki 30 / web 129) | ✅ 147 | ✅ 143 | ✅ 105 (Llama-3.1-8B) | ✅ 111 (121 pass −10 dup) | ✅ **100** | ✅ **6/6 models** |
| **A01-CC-01** | ✅ 143 (SANSKRITI 1 / Wiki 14 / web 128) | ✅ 134 | ✅ 134 | ✅ 102 (Llama-3.1-8B) | ✅ 113 (118 pass −5 dup) | ✅ **100** | ✅ **6/6 models** |

**A01-SS-01 notes:** 100% strict-clean, state-balanced (Kerala 39 / TN 24 / Karnataka 16 / AP 14 / Telangana 7), 0 provenance gaps. Per-model scores appended in `data/final/iccd_A01-SS-01.json`. Cross-model: RLHF pairs preserve/sharpen binding (corr 0.86–0.91); **Sarvam-M (Indian fine-tune) weakens it (Δ +1.36, corr 0.36) — candidate rewrite**; tokenizer gate passed. Details: `docs/QUALITY.md`, `docs/CROSSMODEL.md`.

**A01-NN-01 / A01-EE-01 notes (released).** Built to the A01-SS-01 standard; Tier-1.5 Claude verification was run *before* the model gate (the gate needs AWS/Babel), so the verified pool was over-provisioned, then gated and selected to 100.
- **A01-NN-01 (North):** funnel 208 cand → 197 pairs → 155 filtered → 126 Tier-1.5 verified → **119 base-Llama gate (ΔL>1.0)** → **100 final** (token 50/30/20 hit exactly; states Rajasthan 30 / Uttarakhand 20 / Punjab 17 / UP 10 / Ladakh 10 / Haryana 10 / Delhi 3). 0 provenance gaps. **J&K, Himachal Pradesh, Chandigarh excluded by the F1 ≤3-token target cap** (Chandigarh shares its festivals with Punjab/Haryana as a UT).
- **A01-EE-01 (East):** funnel 214 cand → 209 pairs → 168 filtered → 121 verified → **112 gate** → **100 final** (token 7/77/16 — 2-token-heavy, the Axis-A state↔token correlation; states Odisha 14 / WB 13 / Assam 12 / Nagaland 12 / Tripura 10 / Manipur 9 / Meghalaya 9 / Bihar 7 / Mizoram 7 / Sikkim 7). 0 provenance gaps. **Arunachal Pradesh and Jharkhand excluded by the F1 token cap.**
- Verification caught films, rivers, an island, deities, a cloth (Gamosa), a radio programme, college/tech fests, modern tourism "Mahotsav"s, and mis-attributions (Bundeli Utsav→MP not UP; Lai Haraoba→Manipur not Assam) — all with sources. Corruptors 100% cross-region/well-formed; no leaks; deduped.
- **Cross-model (6/6 models, appended to the release JSONs):** clean-RLHF pairs preserve/sharpen binding — Llama-3.1-8B corr 0.94/0.90 (N/E), Gemma-2-9B corr 0.87/0.89 with aligned *sharper*. **Mistral→Sarvam-M (Indian SFT+RLVR) weakens binding — corr 0.60 (North) / 0.50 (East), Δ +0.53 / +0.96 — the rewrite-direction signal, stronger in East than North (a regional-selectivity hint).** AWS instance + SG torn down; Babel wiped. Details: `docs/QUALITY.md`, `docs/CROSSMODEL.md`.

**A01-WW-01 / A01-CC-01 notes (released, wave 3 — the two festival-thin regions).** Same `scripts/build_cell.py` chain; SANSKRITI is thin here (West 132 / Central 104), so the build leaned heavily on the **web tier**: a structured-output web-research Workflow (one general-purpose agent per state×scope; every festival web-verified with a logged source URL; `distinctive=true` only). Claude Tier-1.5 ran before the Stage-8 gate, then a fuzzy-dup pass dropped spelling/qualifier variants (`scripts/apply_verdicts.py`).
- **A01-WW-01 (West):** funnel 164 cand (SANSKRITI 5 / Wiki 30 / web 129) → 147 pairs → 143 F1-F7 → 121 Claude-pass → **111 after −10 fuzzy-dup** → **105 base-Llama gate (ΔL>1.0)** → **100 final** (Maharashtra 40 / Gujarat 30 / Goa 30). Token mix **1/2/3 = 100/0/0** — all three eligible West states are single-token, so the cell is a clean 1-token stratum (recorded). Provenance 72 Wikipedia-oldid + 28 web-URL, 0 gaps. **Dadra & Nagar Haveli & Daman & Diu excluded by the F1 cap (13 tokens).**
- **A01-CC-01 (Central):** funnel 143 cand (SANSKRITI 1 / Wiki 14 / web 128) → 134 pairs → 134 F1-F7 → 118 Claude-pass → **113 after −5 fuzzy-dup** → **102 base-Llama gate** → **100 final** (Madhya Pradesh 58 / Chhattisgarh 42). Token mix **3/5 = 58/42**. Provenance 54 Wikipedia-oldid + 46 web-URL, 0 gaps. **F1 design exception:** Central's only short-enough states are Madhya Pradesh (3 tok) and Chhattisgarh — which tokenizes to **5**, over the F1 ≤3-token cap. Dropping it would collapse Central to a single state, so Chhattisgarh is kept as a documented exception (`config.F1_TOKEN_EXCEPTIONS`); its items form a 5-token stratum. The ΔL gate and Tier-1.5 verification are unaffected by target length. (User-approved 2026-05-31.)
- Verification caught modern govt "Mahotsav/Utsav" tourism brandings (Bhojpur Utsav, Bhoramdeo Mahotsav), district-name leakage ("Bastar"→Chhattisgarh), EDM/arts events (Sunburn, Serendipity), and fuzzy variants (Zatra/Jatra, Madhavpur Ghed/Fair, Akti/Akti Tihar). Corruptors cross-region and well-formed; no leaks.
- **Cross-model (6/6 models, appended to the release JSONs):** clean-RLHF preserves binding — Llama-3.1-8B corr **0.89** (W) / **0.86** (C); **Gemma-2-9B aligned is *sharper* (corr 0.91 / 0.92, Δ −2.65 / −2.11)**, the same instruction-tuning-sharpens pattern as N/E. **Mistral→Sarvam-M (Indian SFT+RLVR) weakens binding most — corr 0.61 (West) / 0.50 (Central); Central ties East (0.50) for the weakest, reinforcing the regional-selectivity hint.** AWS instance + SG torn down; Babel token/dir/cache wiped.

## Sourcing-ease hints (SANSKRITI festival inventory, Stage 1)
A01 Festivals by region: **South 323, East 311, North 268, West 132, Central 104**. East/North are data-rich (easy next builds); West/Central thinner; Northeast (within East) is Wikipedia-thin → leans on the web tier. (A02/A03 sub-concepts map to other SANSKRITI attributes — Dance_and_Music, Art, Religion, History, Medicine — with some gaps that lean on Wikipedia/web; see `docs/PIPELINE.md`.)

## Mapping notes for the not-yet-built axes
- **A02 (Flattening)** uses description-based counterfactuals + a closed indigenous→generic vocabulary (plan §7.4); SANSKRITI `Dance_and_Music` covers A02-01/02, `Art` covers A02-03; **A02-04 Architecture** has no SANSKRITI attribute → Wikipedia/web.
- **A03 (Sensitive)** description-based, reversed sign; SANSKRITI `Religion`→A03-02, `History`→A03-03, `Medicine`→A03-04; **A03-01 Caste** has no SANSKRITI attribute → Wikipedia/web.
