"""Flexible input loader. Accepts:

- a local JSONL path (one `{"id", "text"}` object per line)
- a local JSON list file
- a HuggingFace dataset identifier (`hf:<dataset>[:<split>]`, e.g. `hf:Anthropic/hh-rlhf:train`)

Normalizes every source into a list of `{"id", "text"}` dicts so the rest of the
pipeline doesn't care where the data came from."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [_normalize_row(r, i) for i, r in enumerate(rows)]


def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return [_normalize_row(r, i) for i, r in enumerate(data)]


def _load_hf(spec: str) -> list[dict]:
    # spec like "hf:dataset_name" or "hf:dataset_name:split"
    parts = spec.removeprefix("hf:").split(":")
    name = parts[0]
    split = parts[1] if len(parts) > 1 else "train"
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "`hf:` spec requires `pip install datasets`; add it to requirements.txt if needed."
        ) from e
    ds = load_dataset(name, split=split)
    return [_normalize_row(r, i) for i, r in enumerate(ds)]


def _normalize_row(row: dict | str, idx: int) -> dict:
    if isinstance(row, str):
        return {"id": f"r{idx}", "text": row}
    row = dict(row)
    if "text" not in row:
        for k in ("content", "body", "message", "utterance", "prompt"):
            if k in row:
                row["text"] = row[k]
                break
    if "id" not in row:
        row["id"] = f"r{idx}"
    return {"id": str(row["id"]), "text": row.get("text", "")}


def load(source: str, *, limit: int | None = None) -> list[dict]:
    """Load rows from `source` and return a list of `{id, text}` dicts."""
    if source.startswith("hf:"):
        rows = _load_hf(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        if path.suffix == ".jsonl":
            rows = _load_jsonl(path)
        elif path.suffix == ".json":
            rows = _load_json(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
