"""Tool (a): exact/normalized lookup against configs/cultural_map.json.

Entries carry a `source` field: "seed" (human-curated) or "retrieved" (auto-added
from Part 2's web+llm path via `append_entry`). Lookup treats both identically —
`source` is metadata for auditing the dictionary."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

CULTURAL_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "cultural_map.json"
RETRIEVED_CATEGORY = "retrieved"


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@lru_cache(maxsize=1)
def _load_index() -> dict[str, dict]:
    raw = json.loads(CULTURAL_MAP_PATH.read_text())
    index: dict[str, dict] = {}
    for category, entries in raw.items():
        for e in entries:
            key = _normalize(e["en"])
            index[key] = {
                "en": e["en"],
                "ko": e["ko"],
                "type": e.get("type", category.rstrip("s")),
                "notes": e.get("notes", ""),
                "approximate": bool(e.get("approximate", False)),
                "keep": bool(e.get("keep", False)),
                "source": e.get("source", "seed" if category != RETRIEVED_CATEGORY else "retrieved"),
            }
    return index


def lookup(term: str) -> dict | None:
    """Return normalized entry dict if `term` matches, else None."""
    return _load_index().get(_normalize(term))


def all_entries() -> list[dict]:
    return list(_load_index().values())


def append_entry(
    term: str,
    ko: str,
    *,
    ref_type: str = "other",
    notes: str = "",
) -> bool:
    """Append a retrieved mapping to cultural_map.json under the `retrieved` category.

    Idempotent: if the term already exists in the map, does nothing and returns False.
    Clears the in-memory cache so subsequent lookups see the new entry.
    """
    term = term.strip()
    ko = ko.strip()
    if not term or not ko:
        return False
    if lookup(term) is not None:
        return False

    raw = json.loads(CULTURAL_MAP_PATH.read_text())
    raw.setdefault(RETRIEVED_CATEGORY, [])
    raw[RETRIEVED_CATEGORY].append({
        "en": term,
        "ko": ko,
        "type": ref_type,
        "notes": notes,
        "source": "retrieved",
    })
    CULTURAL_MAP_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    )
    _load_index.cache_clear()
    return True
