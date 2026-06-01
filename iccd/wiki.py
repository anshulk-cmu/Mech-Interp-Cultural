"""English Wikipedia MediaWiki API client.

Captures revision id (oldid) + access timestamp per page for reproducibility
(plan Section 5.3 / 10.3). An on-disk response cache keyed by query params makes
sourcing resumable and rate-limit friendly.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from .config import RESOURCES, WIKI_API, WIKI_USER_AGENT

_CACHE = RESOURCES / "wiki_cache"
_CACHE.mkdir(parents=True, exist_ok=True)
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = WIKI_USER_AGENT
_MIN_INTERVAL = 0.2
_last = [0.0]


def _get(params: dict) -> dict:
    params = {**params, "format": "json"}
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_file = _CACHE / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    # Retry transient drops (RemoteDisconnected / timeouts) with backoff; the on-disk
    # cache means a resumed run re-reads already-fetched pages instantly.
    last_err = None
    for attempt in range(4):
        gap = time.time() - _last[0]
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        try:
            r = _SESSION.get(WIKI_API, params=params, timeout=30)
            _last[0] = time.time()
            r.raise_for_status()
            data = r.json()
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except (requests.RequestException, ValueError) as e:
            last_err = e
            _last[0] = time.time()
            time.sleep(1.0 * (attempt + 1))
    raise last_err


def category_members(category: str, limit: int = 500, with_subcats: bool = False):
    """Article titles in a category (and subcategory names if with_subcats)."""
    arts: list[str] = []
    subs: list[str] = []
    cont: dict = {}
    while True:
        data = _get({"action": "query", "list": "categorymembers",
                     "cmtitle": f"Category:{category}", "cmtype": "page|subcat",
                     "cmprop": "title|type|ns", "cmlimit": limit, **cont})
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("type") == "subcat":
                subs.append(m["title"].split(":", 1)[-1])
            elif m.get("ns", 0) == 0:
                arts.append(m["title"])
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return (arts, subs) if with_subcats else arts


def category_members_recursive(category: str, depth: int = 1, _seen=None) -> set[str]:
    """Article titles in a category and its subcategories up to `depth`."""
    seen = _seen if _seen is not None else set()
    arts, subs = category_members(category, with_subcats=True)
    out = set(arts)
    if depth > 0:
        for sub in subs:
            if sub not in seen:
                seen.add(sub)
                out |= category_members_recursive(sub, depth - 1, seen)
    return out


def page_summary(title: str) -> dict | None:
    """Intro extract + revision id + access timestamp, or None if missing."""
    data = _get({"action": "query", "prop": "extracts|revisions",
                 "exintro": 1, "explaintext": 1, "redirects": 1,
                 "rvprop": "ids", "titles": title})
    for _, p in data.get("query", {}).get("pages", {}).items():
        if "missing" in p or "extract" not in p:
            return None
        return {
            "title": p["title"],
            "extract": p["extract"].strip(),
            "oldid": p.get("revisions", [{}])[0].get("revid"),
            "url": "https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_"),
            "accessed_utc": datetime.now(timezone.utc).isoformat(),
        }
    return None


def _opensearch(query: str):
    data = _get({"action": "opensearch", "search": query, "limit": 1,
                 "namespace": 0, "redirects": "resolve"})
    if isinstance(data, list) and len(data) > 1 and data[1]:
        return data[1][0]
    return None


def resolve_and_summary(title: str) -> dict | None:
    """page_summary, falling back to a Wikipedia opensearch title resolve (for web-tier items)."""
    s = page_summary(title)
    if s:
        return s
    alt = _opensearch(title)
    if alt and alt.lower() != title.lower():
        return page_summary(alt)
    return None
