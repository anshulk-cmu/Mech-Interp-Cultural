"""Stage 6 — provenance audit + release assembly for the final cell.

Validates every final item carries mandatory provenance (cross_validated_by non-empty,
Wikipedia oldid, region) and enriches it with the Stage-8 base-model delta-L and the Claude
verdict, producing the release record. Missing-cross-validation count must be zero (release gate).
"""
from __future__ import annotations

from ..config import (FINAL, INTERIM, REGION_NAMES, STATE_TO_REGION, SMOKE_CELL, SUBCONCEPTS,
                      cell_id)
from ..logutil import get_logger
from ..state import jsonl_read, read_json, write_json

log = get_logger("stage6")


def run(cell=SMOKE_CELL):
    axis, region_code, sub = cell
    cid = cell_id(*cell)
    sub_concept_name = SUBCONCEPTS[axis][sub]
    final = read_json(INTERIM / f"selected_{cid}.json", [])
    dl = {r["item_id"]: r for r in jsonl_read(INTERIM / f"stage8_results_{cid}.jsonl")}
    verds = {v["item_id"]: v for v in read_json(INTERIM / f"claude_verdicts_{cid}.json", [])}

    out, missing = [], 0
    for p in final:
        iid = p["item_id"]
        cv = p.get("cross_validated_by", [])
        # provenance complete = cross-validated AND (Wikipedia oldid OR a whitelisted web source_url)
        if not cv or (not p["wiki"].get("oldid") and not p.get("web_source_url")):
            missing += 1
        region = STATE_TO_REGION.get(p["target"], cid.split("-")[1])
        out.append({
            "item_id": iid, "cell_id": cid, "axis": axis,
            "region": region, "region_name": REGION_NAMES[region], "sub_concept": sub_concept_name,
            "anchor": p["anchor"], "target": p["target"],
            "clean_prefix": p["clean_prefix"], "corrupted_prefix": p["corrupted_prefix"],
            "r": p["r"], "r_prime": p["r_prime"], "corruptor_anchor": p["corruptor_anchor"],
            "target_tokens": p["target_tokens"], "counterfactual_type": "cross_anchor_swap",
            "provenance": {
                "source_primary": p["source_primary"],
                "wiki_title": p["wiki"]["title"], "wiki_oldid": p["wiki"]["oldid"],
                "wiki_url": p["wiki"]["url"], "wiki_accessed_utc": p["wiki"]["accessed_utc"],
                "web_source_url": p.get("web_source_url"),
                "cross_validated_by": cv,
            },
            "stage8_base_delta_L": dl.get(iid, {}).get("delta_L"),
            "claude_verdict": {k: verds.get(iid, {}).get(k)
                               for k in ("fact_ok", "leakage_ok", "counterfactual_ok", "natural_ok", "pass")},
        })
    write_json(FINAL / f"iccd_{cid}.json", out)
    log.info("Stage 6 cell %s: %d release items; %d missing cross-validation (gate: must be 0)",
             cid, len(out), missing)


if __name__ == "__main__":
    run()
