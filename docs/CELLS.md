# ICCD-6K — 60-Cell Status Tracker

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

60 cells = **3 axes × 5 regions × 4 sub-concepts**, 100 items each (6,000 total).
Cell id = `A0{axis}-{REGION}-0{sub}`, e.g. `A01-SS-01`. Regions: NN North · SS South · EE East · WW West · CC Central.

**Legend:** ✅ done · 🟡 in progress · ⬜ not started.
**Per-cell stages:** source → STR pairs → F1-F7 filters → base-model gate → Claude verify → final 100 → 6-model cross-model scoring.

**Summary: 1 / 60 built (A01-SS-01).** SANSKRITI festival inventory by region (from Stage 1) noted where known, to gauge sourcing ease.

---

## Axis A01 — Regional Specificity  (target answer = the state)

| Sub-concept | NN North | SS South | EE East | WW West | CC Central |
|---|---|---|---|---|---|
| 01 Festivals | ⬜ A01-NN-01 | ✅ A01-SS-01 | ⬜ A01-EE-01 | ⬜ A01-WW-01 | ⬜ A01-CC-01 |
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

**A01-SS-01 notes:** 100% strict-clean, state-balanced (Kerala 39 / TN 24 / Karnataka 16 / AP 14 / Telangana 7), 0 provenance gaps. Per-model scores appended in `data/final/iccd_A01-SS-01.json`. Cross-model: RLHF pairs preserve/sharpen binding (corr 0.86–0.91); **Sarvam-M (Indian fine-tune) weakens it (Δ +1.36, corr 0.36) — candidate rewrite**; tokenizer gate passed. Details: `docs/QUALITY.md`, `docs/CROSSMODEL.md`.

## Sourcing-ease hints (SANSKRITI festival inventory, Stage 1)
A01 Festivals by region: **South 323, East 311, North 268, West 132, Central 104**. East/North are data-rich (easy next builds); West/Central thinner; Northeast (within East) is Wikipedia-thin → leans on the web tier. (A02/A03 sub-concepts map to other SANSKRITI attributes — Dance_and_Music, Art, Religion, History, Medicine — with some gaps that lean on Wikipedia/web; see `docs/PIPELINE.md`.)

## Mapping notes for the not-yet-built axes
- **A02 (Flattening)** uses description-based counterfactuals + a closed indigenous→generic vocabulary (plan §7.4); SANSKRITI `Dance_and_Music` covers A02-01/02, `Art` covers A02-03; **A02-04 Architecture** has no SANSKRITI attribute → Wikipedia/web.
- **A03 (Sensitive)** description-based, reversed sign; SANSKRITI `Religion`→A03-02, `History`→A03-03, `Medicine`→A03-04; **A03-01 Caste** has no SANSKRITI attribute → Wikipedia/web.
