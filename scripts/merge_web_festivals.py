"""Merge a web-research Workflow's festival output into data/resources/web_festivals.json.

Takes the raw festival list returned by the web-research Workflow (one dict per
festival: anchor/target/source_url/wikipedia_title/distinctive/evidence/cell), keeps
distinctive=true entries, ASCII-folds + cleans the anchor, dedups (case-insensitive,
and against anchors already sourced in candidates_raw_<cell>.jsonl), then writes the
per-cell lists into web_festivals.json WITHOUT disturbing other cells' keys.

Usage:  python scripts/merge_web_festivals.py data/interim/web_research_raw_wwcc.json
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from iccd.config import REGIONS, STATE_TO_REGION, WEB_FESTIVALS, cell_id
from iccd.state import read_json, write_json

INTERIM = Path("data/interim")
_PAREN = re.compile(r"\s*\([^)]*\)")
_MOJIBAKE = re.compile(r"[ÃÂ€ƒ½¼Å�]")


def _clean(name: str):
    """ASCII-fold + strip to a natural English anchor; None if unusable."""
    name = _PAREN.sub("", str(name)).strip().strip(".?!'\"’ ").strip()
    if _MOJIBAKE.search(name):
        return None
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()
    name = re.sub(r"\s+", " ", name)
    if len(name) < 3 or len(name.split()) > 5:
        return None
    return name


def _existing_anchor_keys(cid: str) -> set:
    p = INTERIM / f"candidates_raw_{cid}.jsonl"
    keys = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["anchor"].lower())
    return keys


def merge(raw_path: str):
    raw = json.load(open(raw_path, encoding="utf-8"))
    festivals = raw["festivals"] if isinstance(raw, dict) else raw

    web = read_json(WEB_FESTIVALS, {})
    # Append-safe + idempotent: preserve any entries already curated for the cell, and dedup
    # new ones against BOTH those and anchors already in candidates_raw_<cell>.jsonl.
    new_by_cell: dict[str, list] = {}
    seen: dict[str, set] = {}
    dropped = {"not_distinctive": 0, "bad_anchor": 0, "bad_state": 0, "dup": 0}

    for f in festivals:
        if not f.get("distinctive"):
            dropped["not_distinctive"] += 1
            continue
        state = str(f.get("target", "")).strip()
        if state not in STATE_TO_REGION:
            dropped["bad_state"] += 1
            continue
        cid = f.get("cell") or cell_id("A01", STATE_TO_REGION[state], "01")
        anchor = _clean(f.get("anchor", ""))
        if not anchor:
            dropped["bad_anchor"] += 1
            continue
        if cid not in seen:
            seen[cid] = _existing_anchor_keys(cid) | {e["anchor"].lower() for e in web.get(cid, [])}
        k = anchor.lower()
        if k in seen[cid]:
            dropped["dup"] += 1
            continue
        seen[cid].add(k)
        entry = {"anchor": anchor, "target": state, "source_url": f.get("source_url", "")}
        wt = (f.get("wikipedia_title") or "").strip()
        if wt:
            entry["wikipedia_title"] = wt
        new_by_cell.setdefault(cid, []).append(entry)

    for cid, entries in sorted(new_by_cell.items()):
        web[cid] = web.get(cid, []) + entries
        states = {}
        for e in web[cid]:
            states[e["target"]] = states.get(e["target"], 0) + 1
        print(f"{cid}: +{len(entries)} new -> {len(web[cid])} web festivals  {states}")
    write_json(WEB_FESTIVALS, web)
    print(f"dropped: {dropped}")
    print(f"wrote {WEB_FESTIVALS}")


if __name__ == "__main__":
    merge(sys.argv[1])
