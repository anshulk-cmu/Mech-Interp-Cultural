"""Apply a Tier-1.5 verify-Workflow result to a cell: save verdicts, drop fuzzy-dup
variants, and regenerate stage8_input_<cell>.json = the deduped Claude-pass batch.

Fuzzy-dup = two PASS items with the same target whose anchors name the same festival
under a spelling/qualifier variant (e.g. "Ambaji Fair" vs "Bhadarvi Poonam Ambaji Fair",
"Gol Gadhedo" vs "Gol Gadhedo Fair"). Detected on the non-generic ("core") anchor tokens:
either the despaced core strings are equal, or token-Jaccard >= 0.5 sharing a >=5-char
proper-noun token. The first/shortest anchor in a group is kept; the rest are marked
pass=false (reason fuzzy_dup_of:<id>) so finalize_cell and the gate agree.

Usage:  python scripts/apply_verdicts.py A01-WW-01 [data/interim/claude_verdicts_A01-WW-01.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from iccd.config import INTERIM
from iccd.state import read_json, write_json

_GENERIC = {"fair", "mela", "melo", "festival", "festivals", "utsav", "mahotsav", "yatra",
            "jatra", "zatra", "urs", "tihar", "parva", "parab", "poonam", "purnima", "no",
            "the", "of", "ka", "ki", "maha", "shri", "shree"}


def _core(anchor: str):
    toks = [t.strip(".,'\"-").lower() for t in anchor.split()]
    return [t for t in toks if t and t not in _GENERIC]


def _is_dup(a_core, b_core) -> bool:
    if not a_core or not b_core:
        return False
    if "".join(sorted(a_core)) == "".join(sorted(b_core)):
        return True
    sa, sb = set(a_core), set(b_core)
    inter = sa & sb
    jac = len(inter) / len(sa | sb)
    return jac >= 0.5 and any(len(t) >= 5 for t in inter)


def apply(cid: str, verdicts_path: str | None = None):
    vp = Path(verdicts_path) if verdicts_path else INTERIM / f"claude_verdicts_{cid}.json"
    verds = json.load(open(vp, encoding="utf-8"))
    pairs = {p["item_id"]: p for p in read_json(INTERIM / f"pairs_filtered_{cid}.json", [])}

    passed = [v for v in verds if v.get("pass") and v["item_id"] in pairs]
    # group fuzzy-dups per target; keep the shortest anchor as representative
    by_tgt: dict[str, list] = {}
    for v in passed:
        by_tgt.setdefault(pairs[v["item_id"]]["target"], []).append(v)

    dup_loser = {}  # item_id -> kept item_id
    for tgt, vs in by_tgt.items():
        vs.sort(key=lambda v: len(pairs[v["item_id"]]["anchor"]))
        kept = []  # (item_id, core)
        for v in vs:
            core = _core(pairs[v["item_id"]]["anchor"])
            hit = next((kid for kid, kc in kept if _is_dup(core, kc)), None)
            if hit:
                dup_loser[v["item_id"]] = hit
            else:
                kept.append((v["item_id"], core))

    for v in verds:
        if v["item_id"] in dup_loser:
            v["pass"] = False
            v["reason"] = f"fuzzy_dup_of:{dup_loser[v['item_id']]} | " + v.get("reason", "")
    write_json(vp, verds)

    keep_ids = [v["item_id"] for v in passed if v["item_id"] not in dup_loser]
    batch = [{"item_id": iid, "clean_prefix": pairs[iid]["clean_prefix"],
              "corrupted_prefix": pairs[iid]["corrupted_prefix"], "target": pairs[iid]["target"],
              "r": pairs[iid]["r"], "r_prime": pairs[iid]["r_prime"],
              "target_tokens": pairs[iid]["target_tokens"]} for iid in keep_ids]
    write_json(INTERIM / f"stage8_input_{cid}.json", batch)

    nstate = Counter(pairs[i]["target"] for i in keep_ids)
    print(f"{cid}: verdicts {len(verds)} | pass {len(passed)} | fuzzy-dups dropped {len(dup_loser)} "
          f"| stage8_input {len(batch)}")
    print(f"  state dist: {dict(nstate)}")
    if dup_loser:
        print("  dropped fuzzy-dups:")
        for lid, kid in dup_loser.items():
            print(f"    '{pairs[lid]['anchor']}'  ~=  '{pairs[kid]['anchor']}'")


if __name__ == "__main__":
    apply(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
