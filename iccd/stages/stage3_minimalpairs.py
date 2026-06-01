"""Stage 3 — Axis A STR cross-anchor-swap minimal pairs.

Clean prefix names the cell's cultural anchor (festival / textile / dish / ritual,
per sub-concept) -> target state r. The corrupted twin names a same-sub-concept anchor
from a DIFFERENT region -> r' (so r' != r by construction). Both share one fixed relation
frame (config.RELATION_TEMPLATES[axis, sub]), so only the swapped anchor differs (STR),
and the swapped anchor is token-length matched (relax +/-1, logged). Deterministic under
the counterfactual-audit seed. Leakage (F2/F3) and corruption checks are Stage 4.
"""
from __future__ import annotations

import hashlib
import random
import sys

from ..config import CORRUPTOR_BANK, INTERIM, SEEDS, SMOKE_CELL, cell_id, relation_template
from ..logutil import get_logger
from ..state import jsonl_read, read_json, write_json
from ..tok import n_tokens, target_token_len

log = get_logger("stage3")


def _corruptor_pool(clean_region, sub):
    """Clean, different-region, SAME-sub-concept anchors from the reusable corruptor bank
    (STR cross-anchor swap). The bank is keyed by sub-concept then region; a same-sub-concept
    corruptor is mandatory (a textile, not a festival, for A01-02) or the corrupted prefix is
    nonsense under the textile relation frame. Falls back to the legacy flat region-keyed layout
    (treated as sub-concept '01') for back-compat."""
    bank = read_json(CORRUPTOR_BANK, {})
    by_sub = bank.get("by_subconcept")
    region_map = by_sub.get(sub, {}) if by_sub else {k: v for k, v in bank.items() if not k.startswith("_")}
    out = []
    for rc, items in region_map.items():
        if rc.startswith("_") or rc == clean_region:
            continue
        for it in items:
            out.append({"anchor": it["anchor"], "state": it["state"], "region": rc,
                        "ntok": n_tokens(it["anchor"])})
    return out


def run(cell=SMOKE_CELL, force: bool = False):
    axis, region, sub = cell
    cid = cell_id(axis, region, sub)
    src = INTERIM / f"candidates_raw_{cid}.jsonl"
    out_path = INTERIM / f"pairs_{cid}.json"
    if out_path.exists() and not force:
        log.info("present, skip (force=True to rebuild)")
        return

    rng = random.Random(SEEDS["counterfactual_audit"])
    template = relation_template(axis, sub)
    corruptors = _corruptor_pool(region, sub)
    log.info("corruptor pool (other regions, sub %s): %d", sub, len(corruptors))

    pairs = []
    for c in sorted(jsonl_read(src), key=lambda c: c["anchor_key"]):
        anchor, target = c["anchor"], c["target"]
        nt = n_tokens(anchor)
        pick = [x for x in corruptors if x["state"] != target
                and x["anchor"].lower() != anchor.lower() and x["ntok"] == nt]
        relaxed = False
        if not pick:
            pick = [x for x in corruptors if x["state"] != target
                    and x["anchor"].lower() != anchor.lower() and abs(x["ntok"] - nt) <= 1]
            relaxed = True
        if not pick:
            continue
        rng.shuffle(pick)
        corr = pick[0]
        pairs.append({
            "item_id": f"iccd_{cid}_" + hashlib.sha1(c["anchor_key"].encode()).hexdigest()[:10],
            "cell_id": cid, "anchor": anchor, "target": target,
            "r": target, "r_prime": corr["state"],
            "corruptor_anchor": corr["anchor"], "corruptor_region": corr["region"],
            "length_relaxed": relaxed,
            "clean_prefix": template.format(anchor=anchor),
            "corrupted_prefix": template.format(anchor=corr["anchor"]),
            "target_tokens": target_token_len(target),
            "source_primary": c["source_primary"], "wiki": c["wiki"],
            "web_source_url": c.get("web_source_url"),
            "sanskriti_attested": c.get("sanskriti_attested"),
            "wiki_state_confirmed": c.get("wiki_state_confirmed"),
            "cross_validated_by": c["cross_validated_by"],
        })
    write_json(out_path, pairs)
    relaxed_n = sum(1 for p in pairs if p["length_relaxed"])
    log.info("cell %s: %d pairs (%d length-relaxed) -> %s", cid, len(pairs), relaxed_n, out_path.name)


if __name__ == "__main__":
    run(force="--force" in sys.argv)
