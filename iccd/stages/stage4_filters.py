"""Stage 4 — F1-F7 deterministic filters.

Per user directive: hard-reject only clear violations; flag borderline leakage for
Claude (Tier 1.5) rather than dropping it. Hard: F1 target>3 tokens, F5 suffix mismatch,
F6/F7 prefix length, F4 corruption failure, F2 the target STATE name appearing in the
clean prefix (a clear answer leak), F3 a uniquely-state-identifying language. Borderline
(flagged, not dropped): a gazetteer place inside the anchor, a non-unique language.
"""
from __future__ import annotations

import re
import sys

from ..config import (F1_TOKEN_EXCEPTIONS, INTERIM, MAX_TARGET_TOKENS, PREFIX_MAX_TOKENS,
                      PREFIX_MIN_TOKENS, RESOURCES, SMOKE_CELL, SUFFIX_MATCH_TOKENS, cell_id)
from ..logutil import get_logger
from ..state import read_json, write_json
from ..tok import n_tokens, target_token_len, trailing_tokens

log = get_logger("stage4")

# Languages that uniquely identify ONE state -> hard-reject if they appear in the anchor (F3).
# Shared/regional languages (Hindi, Bhojpuri, Bengali, Santali, Nepali, Telugu) are NOT here;
# they only get flagged for Claude. Extended South->North/East for the 60-cell scale-out.
_UNIQUE_LANG = {
    # South
    "Tamil": "Tamil Nadu", "Malayalam": "Kerala", "Kannada": "Karnataka",
    # North
    "Punjabi": "Punjab", "Haryanvi": "Haryana", "Kashmiri": "Jammu and Kashmir",
    "Dogri": "Jammu and Kashmir", "Ladakhi": "Ladakh", "Rajasthani": "Rajasthan",
    "Marwari": "Rajasthan", "Garhwali": "Uttarakhand", "Kumaoni": "Uttarakhand",
    # East
    "Odia": "Odisha", "Assamese": "Assam", "Meitei": "Manipur", "Manipuri": "Manipur",
    "Khasi": "Meghalaya", "Garo": "Meghalaya", "Mizo": "Mizoram", "Kokborok": "Tripura",
    "Lepcha": "Sikkim", "Maithili": "Bihar", "Magahi": "Bihar",
    # West / Central
    "Gujarati": "Gujarat", "Marathi": "Maharashtra", "Konkani": "Goa",
    "Chhattisgarhi": "Chhattisgarh",
}


def _word_in(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.I) is not None


def run(cell=SMOKE_CELL, force: bool = False):
    axis, region, sub = cell
    cid = cell_id(axis, region, sub)
    out_path = INTERIM / f"pairs_filtered_{cid}.json"
    rej_path = INTERIM / f"rejected_{cid}.json"
    if out_path.exists() and not force:
        log.info("present, skip (force=True to rebuild)")
        return

    pairs = read_json(INTERIM / f"pairs_{cid}.json", [])
    gaz = read_json(RESOURCES / "gazetteer.json", {})
    lang = read_json(RESOURCES / "language_map.json", {})

    kept, rejected, counts = [], [], {}
    for p in pairs:
        anchor, target = p["anchor"], p["target"]
        clean, corrupt = p["clean_prefix"], p["corrupted_prefix"]
        flags, code = [], None

        if target_token_len(target) > MAX_TARGET_TOKENS and target not in F1_TOKEN_EXCEPTIONS:
            code = "F1_token_overlong"
        elif not (PREFIX_MIN_TOKENS <= n_tokens(clean) <= PREFIX_MAX_TOKENS):
            code = "F6F7_prefix_length"
        elif trailing_tokens(clean, SUFFIX_MATCH_TOKENS) != trailing_tokens(corrupt, SUFFIX_MATCH_TOKENS):
            code = "F5_suffix_mismatch"
        elif p["r_prime"] == p["r"] or _word_in(corrupt, anchor):
            code = "F4_corruption_failed"
        elif _word_in(clean, target):
            code = "F2_state_name_leak"
        else:
            for lg in lang.get(target, []):
                if _word_in(anchor, lg) and _UNIQUE_LANG.get(lg) == target:
                    code = "F3_unique_language_leak"
                    break
            if code is None:
                for place in gaz.get(target, []):
                    if len(place) >= 4 and _word_in(anchor, place):
                        flags.append(f"F2_place:{place}")
                        break
                for lg in lang.get(target, []):
                    if _word_in(anchor, lg) and _UNIQUE_LANG.get(lg) != target:
                        flags.append(f"F3_language:{lg}")
                        break

        if code:
            rejected.append({"item_id": p["item_id"], "anchor": anchor, "target": target, "code": code})
            counts[code] = counts.get(code, 0) + 1
        else:
            p["flags"] = flags
            kept.append(p)

    write_json(out_path, kept)
    write_json(rej_path, rejected)
    log.info("cell %s: kept %d, rejected %d %s; %d flagged-for-Claude",
             cid, len(kept), len(rejected), counts, sum(1 for k in kept if k["flags"]))


if __name__ == "__main__":
    run(force="--force" in sys.argv)
