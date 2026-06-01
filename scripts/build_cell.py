"""Local per-cell build driver (Stages 2 -> 5 Pass A), reusable across the 60-cell grid.

Runs the same local chain that produced A01-SS-01, for any cell, stopping at the
Stage-8 model gate (which runs on AWS/Babel). Subcommands:

  bootstrap NN EE ...     extend gazetteer (F2) + language_map (F3) to new regions
                          (Wikipedia district/city walk; merges, never overwrites South)
  stage2 A01-NN-01        SANSKRITI + Wikipedia sourcing -> candidates_raw_<cell>.jsonl
  webtier A01-NN-01       inject curated web_festivals.json[cell] -> candidates (+provenance log)
  pairs A01-NN-01         Stage 3 (STR cross-anchor swap) + Stage 4 (F1-F7) + Stage 5 Pass A
  all A01-NN-01           stage2 -> webtier -> pairs (full local chain)

Every stage is idempotent/resumable; `force=True` is used for the deterministic rebuild stages.
"""
from __future__ import annotations

import sys

from iccd.config import REGIONS, RESOURCES, cell_id
from iccd.logutil import get_logger
from iccd.state import read_json, write_json
from iccd.stages import stage0_bootstrap as s0
from iccd.stages import stage2_sourcing as s2
from iccd.stages import stage2b_webtier as s2b
from iccd.stages import stage3_minimalpairs as s3
from iccd.stages import stage4_filters as s4
from iccd.stages import stage5_sample as s5

log = get_logger("build")


def _cell_tuple(cid: str):
    axis, region, sub = cid.split("-")
    return (axis, region, sub)


def bootstrap(region_codes):
    """Extend gazetteer (F2) + language_map (F3) to new regions, merging into existing files."""
    gaz = read_json(RESOURCES / "gazetteer.json", {})
    lang = read_json(RESOURCES / "language_map.json", {})
    states = [st for rc in region_codes for st in REGIONS[rc]]
    new_states = [st for st in states if st not in gaz]
    if new_states:
        log.info("building gazetteer for %d new states: %s", len(new_states), new_states)
        gaz.update(s0.build_gazetteer(new_states))
        write_json(RESOURCES / "gazetteer.json", gaz)
    for st in states:
        lang[st] = s0.LANGUAGE_MAP.get(st, [])
    write_json(RESOURCES / "language_map.json", lang)
    log.info("resources extended: gazetteer %d states, language_map %d states", len(gaz), len(lang))


def stage2(cid: str):
    s2.run(_cell_tuple(cid), force=True)


def webtier(cid: str):
    s2b.run(_cell_tuple(cid))


def pairs(cid: str):
    cell = _cell_tuple(cid)
    s3.run(cell, force=True)
    s4.run(cell, force=True)
    s5.pass_a(cell)


def all_local(cid: str):
    stage2(cid)
    webtier(cid)
    pairs(cid)


def main(argv):
    cmd = argv[0]
    if cmd == "bootstrap":
        bootstrap(argv[1:])
    elif cmd == "stage2":
        stage2(argv[1])
    elif cmd == "webtier":
        webtier(argv[1])
    elif cmd == "pairs":
        pairs(argv[1])
    elif cmd == "all":
        all_local(argv[1])
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
