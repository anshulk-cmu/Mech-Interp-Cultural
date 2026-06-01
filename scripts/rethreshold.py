"""Re-threshold the base-Llama suite scores into per-cell Stage-8 gate files.

The suite scorer runs with --threshold -1000 (score everything). This splits
results_<base_slug>.jsonl by cell (item_ids embed the cell id) and writes
stage8_results_<cell>.jsonl with kept = delta_L > DELTA_L_FLOOR_NATS (1.0), the
Axis-A one-sided gate that finalize_cell.py consumes.

Usage:  python scripts/rethreshold.py A01-WW-01 A01-CC-01 [--base llama31-base]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from iccd.config import DELTA_L_FLOOR_NATS

INTERIM = Path("data/interim")


def run(cells, base_slug="llama31-base", floor=DELTA_L_FLOOR_NATS):
    src = INTERIM / f"results_{base_slug}.jsonl"
    if not src.exists():
        raise SystemExit(f"missing {src} — pull the base-model suite results first")
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    for cid in cells:
        out = INTERIM / f"stage8_results_{cid}.jsonl"
        n = k = 0
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                if cid in r["item_id"]:
                    keep = r["delta_L"] > floor
                    f.write(json.dumps({**r, "kept": keep}) + "\n")
                    n += 1
                    k += int(keep)
        print(f"{cid}: {n} scored, {k} kept (delta_L>{floor}) -> {out.name}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = "llama31-base"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    run(args, base_slug=base)
