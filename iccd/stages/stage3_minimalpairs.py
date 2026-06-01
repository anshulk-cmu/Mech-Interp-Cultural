"""Stage 3 — Axis A STR cross-anchor-swap minimal pairs.

Clean prefix names the cell's festival -> target state r. The corrupted twin names a
same-sub-concept festival from a DIFFERENT region -> r' (so r' != r by construction).
Both share an identical template suffix, so only the swapped anchor differs (STR), and
the swapped anchor is token-length matched (relax +/-1, logged). Deterministic under
the counterfactual-audit seed. Leakage (F2/F3) and corruption checks are Stage 4.
"""
from __future__ import annotations

import hashlib
import random
import sys

from ..config import CORRUPTOR_BANK, INTERIM, SEEDS, SMOKE_CELL, cell_id
from ..logutil import get_logger
from ..state import jsonl_read, read_json, write_json
from ..tok import n_tokens, target_token_len

log = get_logger("stage3")
TEMPLATE = "{anchor} is a festival celebrated in the Indian state of"


def _corruptor_pool(clean_region):
    """Clean, different-region festivals from the reusable corruptor bank (STR cross-anchor swap)."""
    bank = read_json(CORRUPTOR_BANK, {})
    out = []
    for rc, items in bank.items():
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
    corruptors = _corruptor_pool(region)
    log.info("corruptor pool (other regions): %d", len(corruptors))

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
            "clean_prefix": TEMPLATE.format(anchor=anchor),
            "corrupted_prefix": TEMPLATE.format(anchor=corr["anchor"]),
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
