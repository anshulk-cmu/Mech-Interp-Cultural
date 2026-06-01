"""Stage 2b — web-tier supplement: inject curated festivals with logged provenance.

Reads data/resources/web_festivals.json (per cell), fetches each festival's Wikipedia
oldid for provenance + the India-guard, logs the web source to web_sources.jsonl, and
appends to candidates_raw_<cell>.jsonl. Runs after Stage 2; reusable per cell at scale.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from .. import wiki
from ..config import INTERIM, SMOKE_CELL, WEB_FESTIVALS, WEB_SOURCES_LOG, cell_id
from ..logutil import get_logger
from ..state import jsonl_append, jsonl_done_keys, read_json
from .stage2_sourcing import _is_event

log = get_logger("stage2b")


def run(cell=SMOKE_CELL):
    cid = cell_id(*cell)
    web = read_json(WEB_FESTIVALS, {}).get(cid, [])
    out_path = INTERIM / f"candidates_raw_{cid}.jsonl"
    done = jsonl_done_keys(out_path, key="anchor_key")

    added = 0
    for entry in web:
        anchor, state, src_url = entry["anchor"], entry["target"], entry["source_url"]
        k = anchor.lower()
        if k in done or _is_event(anchor):
            continue
        summ = wiki.resolve_and_summary(anchor)
        wiki_ok = summ is not None and (
            "india" in summ["extract"].lower() or state.lower() in summ["extract"].lower())
        now = datetime.now(timezone.utc).isoformat()
        jsonl_append(out_path, {
            "anchor_key": k, "cell_id": cid, "anchor": anchor, "target": state,
            "region": cell[1], "source_primary": "websearch",
            "sanskriti_attested": False, "wiki_state_confirmed": bool(wiki_ok),
            "web_source_url": src_url,
            "wiki": ({"title": summ["title"], "oldid": summ["oldid"], "url": summ["url"],
                      "extract": summ["extract"][:400], "accessed_utc": summ["accessed_utc"]}
                     if summ else {"title": None, "oldid": None, "url": src_url,
                                   "extract": "", "accessed_utc": now}),
            "cross_validated_by": ["wikipedia", "websearch"] if wiki_ok else ["websearch"],
        })
        jsonl_append(WEB_SOURCES_LOG, {"cell_id": cid, "anchor": anchor, "target": state,
                                       "source_url": src_url, "accessed_utc": now})
        added += 1
    log.info("Stage 2b cell %s: +%d web-tier candidates (provenance -> %s)",
             cid, added, WEB_SOURCES_LOG.name)


if __name__ == "__main__":
    run()
