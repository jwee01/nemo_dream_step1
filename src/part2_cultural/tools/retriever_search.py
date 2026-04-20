"""Tool (b): embedding similarity search over cultural_map entries using NeMo
Retriever (`nvidia/llama-3.2-nv-embedqa-1b-v2`). Assumes `index_builder.build()`
has been run to produce a local .npz index."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

INDEX_PATH = Path(__file__).resolve().parents[3] / "configs" / "retriever_index.npz"


def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def _embed(texts: list[str], *, input_type: str) -> np.ndarray:
    model = os.environ.get("RETRIEVER_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
    resp = _client().embeddings.create(
        model=model,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def embed_passages(texts: list[str]) -> np.ndarray:
    return _embed(texts, input_type="passage")


def embed_query(text: str) -> np.ndarray:
    return _embed([text], input_type="query")[0]


@lru_cache(maxsize=1)
def _load_index() -> tuple[np.ndarray, list[dict]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Retriever index not found at {INDEX_PATH}. "
            "Run `python -m src.part2_cultural.index_builder` first."
        )
    data = np.load(INDEX_PATH, allow_pickle=True)
    return data["vectors"], list(data["entries"])


def search(term: str, top_k: int = 3, threshold: float = 0.65) -> list[dict]:
    """Return up to top_k entries whose cosine similarity to `term` >= threshold."""
    vecs, entries = _load_index()
    q = embed_query(term)
    sims = vecs @ q
    order = np.argsort(-sims)[:top_k]
    hits: list[dict] = []
    for i in order:
        score = float(sims[i])
        if score < threshold:
            break
        hit = dict(entries[i])
        hit["score"] = score
        hits.append(hit)
    return hits
