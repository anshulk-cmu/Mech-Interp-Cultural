"""Combine the per-cell releases into the final ICCD-6K dataset + a manifest.

Each cell's Stage 6 writes data/final/iccd_<cellid>.json. This merges every present cell
into data/final/iccd_6k_main.json and writes iccd_6k_manifest.json (cell -> count). item_ids
embed the cell id, so cross-cell collisions can't occur. Works incrementally (combines whatever
cells exist); run a final time once all 60 are built. (Plan release artifact: iccd_6k_main.json.)
"""
import io
import json
import re
from collections import Counter
from pathlib import Path

FINAL = Path("data/final")
CELL = re.compile(r"^iccd_A0[123]-[A-Z]{2}-0[1-4]\.json$")


def main():
    cells = sorted(p for p in FINAL.glob("iccd_A*.json") if CELL.match(p.name))
    items, manifest = [], []
    for p in cells:
        d = json.load(io.open(p, encoding="utf-8"))
        items.extend(d)
        manifest.append({"cell": p.stem.replace("iccd_", ""), "n": len(d)})

    json.dump(items, io.open(FINAL / "iccd_6k_main.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"cells_built": len(cells), "total_items": len(items), "cells": manifest},
              io.open(FINAL / "iccd_6k_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    ids = [it["item_id"] for it in items]
    dups = [k for k, v in Counter(ids).items() if v > 1]
    print(f"combined {len(cells)}/60 cells -> {len(items)} items -> iccd_6k_main.json")
    for m in manifest:
        print(f"  {m['cell']}: {m['n']}")
    print(f"unique item_ids: {len(set(ids))}/{len(ids)}" + (f"  DUPLICATES: {dups}" if dups else ""))


if __name__ == "__main__":
    main()
