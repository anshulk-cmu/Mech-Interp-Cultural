"""Stage 5 — token-balanced stratified sampling around the Stage-8 model gate.

Pass A: carry filtered pairs forward as the Stage-8 scoring batch.
Pass B: from the Stage-8 + Claude survivors, draw the final 100 under the 50/30/20
one/two/three-token balance (random within stratum, seed 42; fallback fill). Axis A
targets are state names, so the realized 1/2/3-token mix is constrained by which states
appear -- we record the realized distribution (plan Stage-5 fallback), never select on delta-L.
"""
from __future__ import annotations

import random
import sys
from collections import Counter

from ..config import (INTERIM, PER_CELL_FINAL, SEEDS, SMOKE_CELL, TOKEN_BALANCE, cell_id)
from ..logutil import get_logger
from ..state import read_json, write_json

log = get_logger("stage5")


def pass_a(cell=SMOKE_CELL):
    cid = cell_id(*cell)
    pairs = read_json(INTERIM / f"pairs_filtered_{cid}.json", [])
    batch = [{"item_id": p["item_id"], "clean_prefix": p["clean_prefix"],
              "corrupted_prefix": p["corrupted_prefix"], "target": p["target"],
              "r": p["r"], "r_prime": p["r_prime"], "target_tokens": p["target_tokens"]}
             for p in pairs]
    out = INTERIM / f"stage8_input_{cid}.json"
    write_json(out, batch)
    log.info("Pass A cell %s: %d items -> %s", cid, len(batch), out.name)


def _round_robin(by_state, want, rng):
    """Pick up to `want` ids, round-robin across states for an even cultural spread."""
    pools = {k: list(v) for k, v in by_state.items()}
    for v in pools.values():
        rng.shuffle(v)
    picked, keys = [], sorted(pools, key=lambda k: -len(pools[k]))
    i = 0
    while len(picked) < want and any(pools.values()):
        k = keys[i % len(keys)]
        if pools[k]:
            picked.append(pools[k].pop())
        i += 1
    return picked


def pass_b(cell=SMOKE_CELL, n=PER_CELL_FINAL):
    cid = cell_id(*cell)
    filtered = {p["item_id"]: p for p in read_json(INTERIM / f"pairs_filtered_{cid}.json", [])}
    survivors = [i for i in read_json(INTERIM / f"stage8_survivors_{cid}.json", []) if i in filtered]
    rng = random.Random(SEEDS["main"])

    # Primary: target token-length strata 50/30/20 (statistical confound control, Zhang & Nanda).
    # Secondary: round-robin across target states within each stratum (cultural spread, fix 4).
    chosen = []
    for k in (1, 2, 3):
        by_state = {}
        for iid in survivors:
            if filtered[iid]["target_tokens"] == k:
                by_state.setdefault(filtered[iid]["target"], []).append(iid)
        chosen += _round_robin(by_state, TOKEN_BALANCE[k], rng)
    if len(chosen) < n:  # fallback fill across remaining survivors; realized split is recorded
        by_state = {}
        for iid in survivors:
            if iid not in chosen:
                by_state.setdefault(filtered[iid]["target"], []).append(iid)
        chosen += _round_robin(by_state, n - len(chosen), rng)
    chosen = chosen[:n]

    final = [filtered[i] for i in chosen]
    write_json(INTERIM / f"selected_{cid}.json", final)
    log.info("Pass B cell %s: %d final; state dist %s; token dist %s", cid, len(final),
             dict(Counter(filtered[i]["target"] for i in chosen)),
             dict(Counter(filtered[i]["target_tokens"] for i in chosen)))


if __name__ == "__main__":
    (pass_b if "--pass-b" in sys.argv else pass_a)()
