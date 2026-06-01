"""Stage 2 — source candidate anchors for a cell (sub-concept aware), cross-validated.

Priority 1: anchors parsed from SANSKRITI questions (already SANSKRITI-attested).
Priority 2: Wikipedia category members, from the sub-concept's category set
(_SUBCONCEPT_SOURCING): per-state categories for festivals, India-level "broad" categories
with extract-based state resolution for textiles. Every kept candidate must clear the
mandatory Wikipedia-AND-SANSKRITI cross-validation (plan Section 5.5 / 9.1). Resumable:
each candidate is appended to a JSONL; a rerun skips anchors already fetched.
"""
from __future__ import annotations

import re
import sys
import unicodedata

from .. import wiki
from ..config import CANDIDATES_PER_CELL, INTERIM, REGIONS, SMOKE_CELL, cell_id
from ..logutil import get_logger
from ..state import jsonl_append, jsonl_done_keys, read_json

log = get_logger("stage2")
SAN_POOL = INTERIM / "sanskriti_pool.json"
SAN_TEXT = INTERIM / "sanskriti_state_text.json"

_Q_PATTERNS = [
    re.compile(r"where is (?:the )?(.+?) famous within", re.I),
    re.compile(r"home to (?:the )?(.+?)\s*\?*\s*$", re.I),
    re.compile(r"^(.+?) is associated to which region", re.I),
]
_PAREN = re.compile(r"\s*\([^)]*\)")
_TAIL = re.compile(r"\s*(festival processions|festivals|festival)\s*$", re.I)
_MOJIBAKE = re.compile(r"[ÃÂ€ƒ½¼Å�]")  # � = replacement char (mangled non-Latin titles)
# words that, left trailing after stripping "Festival", would make a fragment -> keep full title
_FRAGMENT_TAIL = {"film", "dance", "music", "book", "arts", "art", "literary", "comedy",
                  "food", "drama", "theatre", "jazz", "rock", "short", "science"}
# clear non-(traditional-festival) markers excluded at source; nuanced calls left to Claude
_EXCLUDE = re.compile(r"(film festival|international film|music season|music festival|book fair|"
                      r"boat league|premier league|\btournament\b|\bleague\b|literary|comic|"
                      r"biennale|marathon|\bcup\b|championship|\bawards?\b|\btrophy\b)", re.I)
_YEAR = re.compile(r"\b(18|19|20)\d\d\b")
# --- Per-sub-concept Wikipedia sourcing (generalizes the festival-only category lists) ---
# state_cats : "{s}"-templated categories fetched PER STATE -> member attributed to that state.
# extra_cats : per-state language/community categories (same per-state attribution).
# broad_cats : region-agnostic India-level categories fetched ONCE; each member's target state
#              is RESOLVED FROM ITS WIKIPEDIA EXTRACT, restricted to the cell's region (so a
#              shared category can't misattribute). Used where per-state categories are empty
#              on Wikipedia (textiles: per-state "Textile arts of {s}" categories don't exist,
#              but "Textile arts of India"/"Saris"/"Embroidery in India" are rich — probed 2026-06-01).
_SUBCONCEPT_SOURCING = {
    ("A01", "01"): {  # Festivals (released cells; behavior unchanged)
        "state_cats": ["Festivals in {s}", "Hindu festivals in {s}", "Religious festivals in {s}"],
        "extra_cats": {
            # South
            "Tamil Nadu": ["Tamil festivals", "Tamil Hindu festivals"],
            "Karnataka": ["Kannada festivals"],
            "Kerala": ["Malayali festivals", "Hindu festivals in Kerala"],
            "Andhra Pradesh": ["Telugu festivals"],
            "Telangana": ["Telangana festivals"],
            # North / East language-community categories that exist on Wikipedia (probed 2026-05-31);
            # non-existent ones (e.g. "Assamese/Odia festivals") return empty and are harmless.
            "Punjab": ["Punjabi festivals"],
            "West Bengal": ["Bengali festivals", "Bengali Hindu festivals"],
            # West / Central language-community categories (probed 2026-05-31); absent ones are harmless.
            "Maharashtra": ["Marathi festivals", "Marathi Hindu festivals"],
            "Gujarat": ["Gujarati festivals", "Gujarati Hindu festivals"],
            "Goa": ["Goan festivals", "Konkani festivals"],
            "Madhya Pradesh": ["Adivasi festivals"],
        },
        "broad_cats": [],
    },
    ("A01", "02"): {  # Costume & Textile
        # Per-state "Textile arts of {s}" / "Crafts of {s}" categories are empty on en.wikipedia
        # (probed 2026-06-01); kept as future-proof no-ops. The real yield is the broad cats below,
        # resolved to a state via each article's extract. SANSKRITI + the web tier are the workhorses.
        "state_cats": ["Textile arts of {s}", "Crafts of {s}"],
        "extra_cats": {},
        "broad_cats": ["Textile arts of India", "Saris", "Embroidery in India", "Indian clothing"],
    },
}
# Generic tokens skipped when SANSKRITI-attesting an anchor (festival + costume vocabulary, len>=5).
_GENERIC = {"festival", "fair", "jatra", "mela", "utsav", "puja", "pooja", "temple",
            "hindu", "day", "night", "feast", "celebration", "harvest", "annual",
            "saree", "sari", "cotton", "cloth", "shawl", "textile", "weave", "weaving",
            "embroidery", "print", "attire", "costume", "dress", "handloom", "fabric", "border"}


def _is_event(text: str) -> bool:
    """Clear non-traditional-festival markers (film/sport/dated/temple) -> exclude at source."""
    t = text.strip()
    return bool(_EXCLUDE.search(t) or _YEAR.search(t) or t.lower().endswith("temple"))


def _attested(anchor: str, blob: str) -> bool:
    for w in anchor.lower().replace("-", " ").split():
        w = w.strip(".,()")
        if len(w) >= 5 and w not in _GENERIC and w in blob:
            return True
    return False


def _clean(name: str):
    name = _PAREN.sub("", name).strip().strip(".?!'\"’ ").strip()
    if _MOJIBAKE.search(name):
        return None
    # Fold Latin diacritics to ASCII for natural English-template anchors (Northeast titles
    # often carry â/û/ü); the original diacritic title is still used for the Wikipedia fetch.
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()
    if not name:
        return None
    stripped = _TAIL.sub("", name).strip()
    if stripped and stripped.split()[-1].lower() not in _FRAGMENT_TAIL:
        name = stripped  # safe to drop trailing "Festival" without leaving a fragment
    if len(name) < 3 or len(name.split()) > 5 or " and " in f" {name.lower()} ":
        return None
    return name


def _sanskriti_anchors(items):
    out = {}
    for it in items:
        for pat in _Q_PATTERNS:
            m = pat.search(it["question"])
            if m:
                a = _clean(m.group(1))
                if a:
                    out.setdefault(a.lower(), (a, it["state"]))
                break
    return out


def _states_in_extract(extract: str, states) -> list:
    """Region states whose name appears (word-boundary) in a Wikipedia extract."""
    el = extract.lower()
    return [s for s in states if re.search(r"\b" + re.escape(s.lower()) + r"\b", el)]


def _wiki_anchors(axis, sub, region):
    """Candidate (anchor, state, title) map for a cell, from sub-concept-matched categories.

    state_cats/extra_cats are per-state (attribution = the loop state); broad_cats are India-level
    and fetched once, with each member's state RESOLVED from its extract among the region's states
    (skip if 0 or >1 region-states match -> ambiguous, left to SANSKRITI/web/Claude)."""
    states = REGIONS[region]
    cfg = _SUBCONCEPT_SOURCING.get((axis, sub))
    if cfg is None:
        raise KeyError(f"no Wikipedia sourcing config for ({axis}, {sub}) — add it to _SUBCONCEPT_SOURCING")
    out = {}
    for st in states:
        cats = [c.format(s=st) for c in cfg["state_cats"]] + cfg["extra_cats"].get(st, [])
        for cat in cats:
            for title in wiki.category_members_recursive(cat, depth=2):
                a = _clean(title)
                if a:
                    out.setdefault(a.lower(), (a, st, title))
    for cat in cfg.get("broad_cats", []):
        for title in wiki.category_members_recursive(cat, depth=1):
            a = _clean(title)
            if not a or a.lower() in out:
                continue
            summ = wiki.page_summary(title)
            if summ is None:
                continue
            hits = _states_in_extract(summ["extract"], states)
            if len(hits) == 1:                       # single region-state named -> attribute it
                out.setdefault(a.lower(), (a, hits[0], title))
    return out


def run(cell=SMOKE_CELL, target=CANDIDATES_PER_CELL, force: bool = False):
    axis, region, sub = cell
    cid = cell_id(axis, region, sub)
    out_path = INTERIM / f"candidates_raw_{cid}.jsonl"
    if force and out_path.exists():
        out_path.unlink()
    done = jsonl_done_keys(out_path, key="anchor_key")

    pool = read_json(SAN_POOL, {})
    state_text = read_json(SAN_TEXT, {})
    san = _sanskriti_anchors(pool.get(cid, []))
    wk = _wiki_anchors(axis, sub, region)

    cand = {k: (a, st, "sanskriti", a) for k, (a, st) in san.items()}
    for k, (a, st, title) in wk.items():
        cand.setdefault(k, (a, st, "wikipedia", title))
    log.info("cell %s: %d sanskriti + %d wiki anchors -> %d unique; %d already done",
             cid, len(san), len(wk), len(cand), len(done))

    kept = len(done)
    tried = 0
    for k, (anchor, state, src, title) in cand.items():
        if kept >= target:
            break
        if k in done:
            continue
        tried += 1
        if _is_event(anchor) or _is_event(title):
            continue
        summ = wiki.page_summary(title)
        if summ is None:
            continue
        extract_l = summ["extract"].lower()
        # India-guard: the page must be about India / the state (kills off-topic matches like the
        # UK "The Boat Race"). Then keep if SANSKRITI-attested OR the intro confirms the
        # festival<->state fact; nuanced festival-vs-event calls remain Claude's job (Tier 1.5).
        if "india" not in extract_l and state.lower() not in extract_l:
            continue
        san_ok = (src == "sanskriti") or _attested(anchor, state_text.get(state, ""))
        wiki_state_ok = state.lower() in extract_l
        if not (san_ok or wiki_state_ok):
            continue
        jsonl_append(out_path, {
            "anchor_key": k, "cell_id": cid, "anchor": anchor, "target": state,
            "region": region, "source_primary": src,
            "sanskriti_attested": san_ok, "wiki_state_confirmed": wiki_state_ok,
            "wiki": {"title": summ["title"], "oldid": summ["oldid"], "url": summ["url"],
                     "extract": summ["extract"][:400], "accessed_utc": summ["accessed_utc"]},
            "cross_validated_by": ["wikipedia"] + (["sanskriti"] if san_ok else []),
        })
        kept += 1
    log.info("cell %s: %d candidates (tried %d new) -> %s", cid, kept, tried, out_path.name)


if __name__ == "__main__":
    run(force="--force" in sys.argv)
