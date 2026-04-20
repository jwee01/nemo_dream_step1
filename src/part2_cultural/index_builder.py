"""Embed every cultural_map entry once and persist as a numpy .npz index."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.part2_cultural.tools.dict_lookup import all_entries
from src.part2_cultural.tools.retriever_search import INDEX_PATH, embed_passages


def _passage_text(entry: dict) -> str:
    notes = entry.get("notes", "")
    base = f"{entry['en']} ({entry['type']})"
    return f"{base}. {notes}" if notes else base


def build() -> Path:
    entries = all_entries()
    texts = [_passage_text(e) for e in entries]
    print(f"[index_builder] embedding {len(texts)} entries via NeMo Retriever...")
    vectors = embed_passages(texts)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, vectors=vectors, entries=np.array(entries, dtype=object))
    print(f"[index_builder] wrote {INDEX_PATH} ({vectors.shape})")
    return INDEX_PATH


if __name__ == "__main__":
    build()
