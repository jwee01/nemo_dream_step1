"""Tool (c): web search fallback via Tavily. Used when dict + retriever miss."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def _is_placeholder(key: str | None) -> bool:
    return not key or key.endswith("...") or key in {"tvly-...", "your-key-here"}


def search(query: str, max_results: int = 3) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("TAVILY_API_KEY")
    if _is_placeholder(api_key):
        return [{"title": f"[web_search disabled] {query}",
                 "content": "TAVILY_API_KEY not set; returning empty.", "url": ""}]
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        res = client.search(
            query=f"Korean cultural equivalent of: {query}",
            search_depth="basic",
            max_results=max_results,
        )
        return [
            {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
            for r in res.get("results", [])
        ]
    except Exception as exc:
        return [{"title": f"[web_search error: {type(exc).__name__}]",
                 "content": str(exc)[:300], "url": ""}]
