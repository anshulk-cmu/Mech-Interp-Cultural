"""Post-Stage-8 finalizer: turn scored candidates into the released 100-item cell.

Run AFTER the AWS/Babel Stage-8 ΔL gate has produced stage8_results_<cell>.jsonl.
Pipeline: survivors = (Claude-Tier-1.5 pass) AND (base-model ΔL > 1.0)  ->  Stage-5 Pass B
(token-balanced 100, seed 42, never ranked on ΔL)  ->  Stage-6 release assembly.

For A01-NN-01 / A01-EE-01 the Claude verification was run BEFORE the gate (the gate needs
GPU), so stage8_input_<cell>.json already contains only the verified pairs; this just
intersects them with the model gate and selects the final 100. Reusable for any cell.

Usage:  python scripts/finalize_cell.py A01-NN-01
"""
from __future__ import annotations

import sys
from collections import Counter

from iccd.config import INTERIM, PER_CELL_FINAL
from iccd.state import jsonl_read, read_json, write_json
from iccd.stages import stage5_sample as s5
from iccd.stages import stage6_provenance as s6


def finalize(cid: str):
    cell = tuple(cid.split("-"))
    pairs = {p["item_id"]: p for p in read_json(INTERIM / f"pairs_filtered_{cid}.json", [])}
    verds = read_json(INTERIM / f"claude_verdicts_{cid}.json", [])
    claude_pass = {v["item_id"] for v in verds if v.get("pass")}

    results = {r["item_id"]: r for r in jsonl_read(INTERIM / f"stage8_results_{cid}.jsonl")}
    if not results:
        raise SystemExit(f"no stage8_results_{cid}.jsonl yet — run the Stage-8 GPU scorer first")
    gate_kept = {iid for iid, r in results.items() if r.get("kept")}

    survivors = [iid for iid in claude_pass if iid in gate_kept and iid in pairs]
    write_json(INTERIM / f"stage8_survivors_{cid}.json", survivors)
    print(f"{cid}: Claude-pass {len(claude_pass)}  AND  gate-kept {len(gate_kept)}  -> survivors {len(survivors)}")
    if len(survivors) < PER_CELL_FINAL:
        print(f"  WARNING: only {len(survivors)} survivors (< {PER_CELL_FINAL}); cell will be short — source more candidates")

    s5.pass_b(cell)                 # -> selected_<cid>.json (token-balanced 100, seed 42)
    s6.run(cell)                    # -> data/final/iccd_<cid>.json (release; 0-missing-XV gate)

    final = read_json(INTERIM / f"selected_{cid}.json", [])
    print(f"{cid}: final {len(final)} | states {dict(Counter(p['target'] for p in final))}"
          f" | tokens {dict(Counter(p['target_tokens'] for p in final))}")


if __name__ == "__main__":
    finalize(sys.argv[1])
