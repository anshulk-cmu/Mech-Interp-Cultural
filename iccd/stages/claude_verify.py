"""Tier 1.5 — Claude in-session verification (prepare + apply).

prepare(): from the Stage-8 survivors, build the batch a Claude Code subagent reviews on
four checks — (1) festival<->state fact valid (flexible second-source, not exact-string),
(2) clean prefix doesn't leak the answer, (3) counterfactual is genuinely different-region
and plausible, (4) both prefixes read naturally. The judgment is produced by Claude Code
subagents (Max subscription, no API key), written to claude_verdicts_<cid>.json.
apply(): merge verdicts -> the model+Claude survivor id list for Stage-5 Pass B.
"""
from __future__ import annotations

from ..config import INTERIM, SMOKE_CELL, cell_id
from ..logutil import get_logger
from ..state import jsonl_read, read_json, write_json

log = get_logger("claude")


def prepare(cell=SMOKE_CELL):
    cid = cell_id(*cell)
    pairs = {p["item_id"]: p for p in read_json(INTERIM / f"pairs_filtered_{cid}.json", [])}
    kept = [r["item_id"] for r in jsonl_read(INTERIM / f"stage8_results_{cid}.jsonl") if r["kept"]]
    batch = []
    for iid in kept:
        p = pairs.get(iid)
        if not p:
            continue
        batch.append({
            "item_id": iid, "anchor": p["anchor"], "target": p["target"],
            "clean_prefix": p["clean_prefix"], "corrupted_prefix": p["corrupted_prefix"],
            "corruptor_anchor": p["corruptor_anchor"], "r_prime": p["r_prime"],
            "wiki_extract": p["wiki"]["extract"][:240], "flags": p.get("flags", []),
        })
    out = INTERIM / f"claude_input_{cid}.json"
    write_json(out, batch)
    log.info("prepared %d Stage-8 survivors for Claude verification -> %s", len(batch), out.name)


def combine(cell=SMOKE_CELL):
    cid = cell_id(*cell)
    merged = []
    for part in sorted(INTERIM.glob("claude_verdicts_part_*.json")):
        merged.extend(read_json(part, []))
    write_json(INTERIM / f"claude_verdicts_{cid}.json", merged)
    log.info("combined %d Claude verdicts; %d pass / %d flag",
             len(merged), sum(v.get("pass") for v in merged), sum(not v.get("pass") for v in merged))


def apply(cell=SMOKE_CELL):
    cid = cell_id(*cell)
    verdicts = read_json(INTERIM / f"claude_verdicts_{cid}.json", [])
    approved = [v["item_id"] for v in verdicts if v.get("pass")]
    write_json(INTERIM / f"stage8_survivors_{cid}.json", approved)
    log.info("Claude approved %d, flagged %d -> stage8_survivors_%s.json",
             len(approved), sum(1 for v in verdicts if not v.get("pass")), cid)
