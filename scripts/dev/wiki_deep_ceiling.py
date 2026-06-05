"""Deep Wikipedia recalibration for the two textile-thin cells A01-WW-02 (West) & A01-CC-02 (Central).

The production Stage-2 sourcing crawls only 4 broad textile categories at depth-1 with
intro-ONLY single-state matching (iccd/stages/stage2_sourcing.py). This is a deliberately
shallow, high-precision pass. To *recalibrate the ceiling* we crawl a much wider set of
textile-domain category roots at depth-2/3, resolve the state from a FULLER extract
(~2000 chars, not just the intro), and keep any article that (a) names exactly one of the
five West/Central target states and (b) reads as a textile. Output is a candidate inventory
grouped by state, with NEW-vs-already-held flagged for West (vs the 57 already verified).

Run:  PYTHONPATH=. python scripts/dev/wiki_deep_ceiling.py
Reads:  data/interim/stage8_input_A01-WW-02.json   (the 57 held West anchors, for dedup)
Writes: data/interim/wiki_deep_ceiling.json
"""
from __future__ import annotations

import json
import re

from iccd import wiki
from iccd.config import INTERIM

# Five eligible target states for the two cells, each -> its cell id.
TARGETS = {
    "Gujarat": "A01-WW-02", "Maharashtra": "A01-WW-02", "Goa": "A01-WW-02",
    "Madhya Pradesh": "A01-CC-02", "Chhattisgarh": "A01-CC-02",
}
WEST = {s for s, c in TARGETS.items() if c == "A01-WW-02"}

# Textile-domain category roots crawled (category, recursion depth). Broad enough to harvest the
# long tail; precision is recovered downstream by the single-state-match + textile-keyword gate.
CATS = [
    ("Textile arts of India", 3),
    ("Saris", 2),
    ("Embroidery in India", 2),
    ("Indian clothing", 2),
    ("Indian silk", 2),
    ("Resist dyeing", 1),
    ("Woven fabrics of India", 2),
    ("Block printing", 1),
    ("Crafts of India", 1),
    ("Geographical indications of India", 1),
    ("Geographical indications in India", 1),
    # per-state craft/textile cats (mostly empty on en.wp per stage2 probe — harmless if so)
    ("Crafts of Gujarat", 2), ("Crafts of Maharashtra", 2), ("Crafts of Goa", 2),
    ("Crafts of Madhya Pradesh", 2), ("Crafts of Chhattisgarh", 2),
    ("Textile arts of Gujarat", 2), ("Textile arts of Maharashtra", 2),
]

TEXTILE_KW = re.compile(
    r"\b(sari|saree|saree|silk|cotton|wool|woven|weav|weave|weaving|embroider|textile|fabric|"
    r"cloth|print|printed|printing|dye|dyed|tie-dye|handloom|loom|shawl|brocade|ikat|"
    r"garment|dupatta|turban|carpet|rug|thread|brocaded|quilt|wrap|drape|attire|costume|"
    r"block-print|kasab|zari|patola|bandhani|batik)\b", re.I)

# names that are clearly not a single textile craft -> drop
_BAD_TAIL = re.compile(r"(film|festival|award|company|brand|museum|list of|company|university|"
                       r"election|district|division|railway|airport|temple|painting)$", re.I)
_PAREN = re.compile(r"\s*\([^)]*\)")


def held_west_anchors() -> set[str]:
    """Normalised base names of the 57 already-verified West anchors (for dedup)."""
    path = INTERIM / "stage8_input_A01-WW-02.json"
    out: set[str] = set()
    if not path.exists():
        return out
    items = json.loads(path.read_text(encoding="utf-8"))
    frame = " is a folk textile tradition"
    for it in items:
        cp = it.get("clean_prefix", "")
        if frame in cp:
            out.add(_norm(cp.split(frame)[0]))
    return out


_SUFFIX = re.compile(r"\s+(saree|sari|sarees|silk|print|printing|printed|embroidery|weave|weaving|"
                     r"cotton|work|shawl|batik|cloth|fabric|art)$", re.I)


def _norm(s: str) -> str:
    s = _PAREN.sub("", str(s or "")).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s).strip()
    s = _SUFFIX.sub("", s).strip()
    return s


def _clean_anchor(title: str) -> str | None:
    name = _PAREN.sub("", title).strip()
    if len(name) < 3 or len(name.split()) > 5:
        return None
    if _BAD_TAIL.search(name):
        return None
    return name


def _states_in(text: str) -> list[str]:
    tl = text.lower()
    return [s for s in TARGETS if re.search(r"\b" + re.escape(s.lower()) + r"\b", tl)]


def full_extract(title: str):
    """Up to ~2000 chars of plaintext (NOT intro-only) + oldid/url, or None."""
    data = wiki._get({"action": "query", "prop": "extracts|revisions",
                      "explaintext": 1, "exchars": 2000, "redirects": 1,
                      "rvprop": "ids", "titles": title})
    for _, p in data.get("query", {}).get("pages", {}).items():
        if "missing" in p or "extract" not in p:
            return None
        return {"title": p["title"], "extract": p["extract"].strip(),
                "oldid": p.get("revisions", [{}])[0].get("revid"),
                "url": "https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_")}
    return None


def main():
    held = held_west_anchors()
    print(f"[held] {len(held)} West anchors loaded for dedup")

    titles: set[str] = set()
    for cat, depth in CATS:
        try:
            members = wiki.category_members_recursive(cat, depth=depth)
        except Exception as e:  # missing category -> empty, keep going
            print(f"[cat] {cat!r}: error {e}")
            continue
        print(f"[cat] {cat!r} (depth {depth}): {len(members)} articles")
        titles |= members
    print(f"[crawl] {len(titles)} unique candidate articles total")

    by_state: dict[str, list] = {s: [] for s in TARGETS}
    seen: set[tuple] = set()
    examined = 0
    for title in sorted(titles):
        anchor = _clean_anchor(title)
        if not anchor:
            continue
        summ = full_extract(title)
        examined += 1
        if not summ:
            continue
        ex = summ["extract"]
        if not TEXTILE_KW.search(ex):
            continue
        sts = _states_in(ex)
        if len(sts) != 1:            # ambiguous / non-target -> skip (matches the build's rule)
            continue
        st = sts[0]
        key = (st, _norm(anchor))
        if key in seen:
            continue
        seen.add(key)
        is_new = not (st in WEST and _norm(anchor) in held)
        by_state[st].append({
            "anchor": anchor, "target": st, "cell": TARGETS[st],
            "wikipedia_title": summ["title"], "oldid": summ["oldid"], "url": summ["url"],
            "already_held_west": (st in WEST and _norm(anchor) in held),
            "is_new": is_new,
            "extract": ex[:300],
        })

    print(f"[resolve] examined {examined} articles with full extract\n")
    out = {"by_state": by_state, "summary": {}}
    print("=" * 64)
    print("DEEP-WIKIPEDIA CEILING RECALIBRATION")
    print("=" * 64)
    for cell, label in [("A01-WW-02", "WEST"), ("A01-CC-02", "CENTRAL")]:
        states = [s for s in TARGETS if TARGETS[s] == cell]
        print(f"\n{label}  ({cell})")
        cell_new = 0
        for s in states:
            items = by_state[s]
            new = [i for i in items if i["is_new"]]
            cell_new += len(new)
            out["summary"][s] = {"total_wiki_textiles": len(items), "new_vs_held": len(new)}
            print(f"  {s:16s}: {len(items):3d} wiki-textiles  ({len(new)} NEW vs held)")
            for i in items:
                tag = "NEW " if i["is_new"] else "held"
                print(f"      [{tag}] {i['anchor']}")
        out["summary"][cell + "_total_new"] = cell_new
        print(f"  -> {cell} candidates NEW from deep Wikipedia: {cell_new}")

    (INTERIM / "wiki_deep_ceiling.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[write] data/interim/wiki_deep_ceiling.json")


if __name__ == "__main__":
    main()
