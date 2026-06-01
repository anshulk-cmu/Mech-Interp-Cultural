"""Stage 1 — SANSKRITI ingest + filter -> per-cell pool + cross-validation index.

SANSKRITI is a content seed, not ground truth (plan Section 5.1). We keep the
region-bearing question types, map attribute->(axis, subconcept), normalize states
to the locked 5-region taxonomy, and build a per-state text blob used later as the
SANSKRITI half of the mandatory Wikipedia-AND-SANSKRITI cross-validation.
"""
from __future__ import annotations

import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from ..config import (ATTRIBUTE_MAP, INTERIM, RAW, SANSKRITI_FILE, SANSKRITI_KEEP_QTYPES,
                      SANSKRITI_REPO, STATE_TO_REGION, cell_id, get_hf_token,
                      normalize_sanskriti_state)
from ..logutil import get_logger
from ..state import write_json

log = get_logger("stage1")
POOL = INTERIM / "sanskriti_pool.json"
STATE_TEXT = INTERIM / "sanskriti_state_text.json"
_XVAL_COLS = ["question", "option1", "option2", "option3", "option4", "answer"]


def _csv_path():
    dest = RAW / "sanskriti"
    p = dest / SANSKRITI_FILE
    if not p.exists():
        log.info("downloading SANSKRITI from %s ...", SANSKRITI_REPO)
        hf_hub_download(repo_id=SANSKRITI_REPO, filename=SANSKRITI_FILE,
                        repo_type="dataset", local_dir=str(dest), token=get_hf_token())
    return p


def run(force: bool = False):
    if POOL.exists() and STATE_TEXT.exists() and not force:
        log.info("outputs present, skipping (force=True to rebuild)")
        return
    df = pd.read_csv(_csv_path())
    log.info("loaded %d rows", len(df))

    df = df[df["question_type"].isin(SANSKRITI_KEEP_QTYPES)].copy()
    log.info("after qtype keep: %d", len(df))

    df["canon_state"] = df["state"].map(normalize_sanskriti_state)
    df = df[df["canon_state"].notna()].copy()
    df["region"] = df["canon_state"].map(STATE_TO_REGION)
    log.info("after state normalize/in-scope: %d", len(df))

    pool: dict[str, list] = {}
    for _, r in df.iterrows():
        m = ATTRIBUTE_MAP.get(str(r["attribute"]))
        if not m:
            continue
        axis, sub = m
        cid = cell_id(axis, r["region"], sub)
        pool.setdefault(cid, []).append({
            "state": r["canon_state"], "region": r["region"], "attribute": str(r["attribute"]),
            "question": str(r["question"]), "answer": str(r["answer"]),
            "question_type": str(r["question_type"]),
            "source_url": str(r.get("short explaination / source link", "")),
        })

    state_text: dict[str, str] = {}
    for st, g in df.groupby("canon_state"):
        texts = []
        for c in _XVAL_COLS:
            texts.extend(g[c].astype(str).str.lower().tolist())
        state_text[st] = " ".join(texts)

    write_json(POOL, pool)
    write_json(STATE_TEXT, state_text)
    log.info("bucketed %d items into %d cells", sum(len(v) for v in pool.values()), len(pool))
    for cid in sorted(pool):
        log.info("  %s: %d", cid, len(pool[cid]))
    log.info("wrote %s, %s", POOL.name, STATE_TEXT.name)


if __name__ == "__main__":
    run(force="--force" in sys.argv)
