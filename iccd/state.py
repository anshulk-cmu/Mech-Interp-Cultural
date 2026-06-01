"""Resumability: atomic JSON outputs + per-item JSONL checkpoints.

Stages are idempotent. Whole-stage outputs go through write_json (atomic replace).
Long per-item loops (Wikipedia fetch, model scoring) append to a JSONL and skip
keys already present on rerun, so an interrupted stage resumes where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def jsonl_done_keys(path: Path, key: str = "item_id") -> set:
    done: set = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)[key])
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def jsonl_append(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
