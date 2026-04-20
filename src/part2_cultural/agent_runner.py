"""Stage 2 orchestrator. Primary path uses NeMo Agent Toolkit's react_agent defined
in `configs/cultural_agent.yaml`; falls back to a direct deterministic implementation
(dict → retriever → web+LLM) when the toolkit is not installed.

Both paths produce the same `MappedRef` objects."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.part2_cultural.tools import dict_lookup, retriever_search, web_search
from src.schemas import CulturalRef, MappedRef

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cultural_agent.yaml"

_WEB_REASONING_PROMPT = (
    "You are localizing an English cultural reference to its closest Korean cultural "
    'equivalent. English term: "{term}" (type: {type}).\n\n'
    "Web search results:\n{results}\n\n"
    "Respond with a single JSON object: "
    '{{"ko": "<Korean term>", "notes": "<one short sentence in Korean or English>"}}. '
    "JSON only, no prose."
)


def _nemotron_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def _web_then_llm(term: str, ref_type: str) -> MappedRef:
    results = web_search.search(term, max_results=3)
    results_block = "\n".join(
        f"- {r.get('title','')}: {r.get('content','')[:300]}" for r in results
    ) or "(no web results)"

    client = _nemotron_client()
    model = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": _WEB_REASONING_PROMPT.format(
                term=term, type=ref_type, results=results_block)},
        ],
        extra_body={"nvext": {"guided_json": {
            "type": "object",
            "properties": {"ko": {"type": "string"}, "notes": {"type": "string"}},
            "required": ["ko"],
        }}},
        temperature=0.2,
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    ko = (payload.get("ko") or "").strip()
    notes = payload.get("notes", "")
    if ko:
        dict_lookup.append_entry(term, ko, ref_type=ref_type, notes=notes)
    return MappedRef(
        term=term,
        ko=ko or term,
        type=ref_type,
        source="web+llm",
        retrieved=True,
        notes=notes,
    )


def _map_one_direct(ref: CulturalRef) -> MappedRef:
    hit = dict_lookup.lookup(ref.term)
    if hit is not None:
        return MappedRef(
            term=ref.term,
            ko=hit["ko"],
            type=hit["type"],
            source="dict",
            retrieved=False,
            notes=hit.get("notes", ""),
        )

    try:
        retrieved = retriever_search.search(ref.term, top_k=1, threshold=0.65)
    except FileNotFoundError:
        retrieved = []
    if retrieved:
        top = retrieved[0]
        return MappedRef(
            term=ref.term,
            ko=top["ko"],
            type=top["type"],
            source="retriever",
            retrieved=False,
            notes=f"cosine={top['score']:.3f}; {top.get('notes','')}".strip("; "),
        )

    return _web_then_llm(ref.term, ref.type)


from src.proxy_patch import apply_proxy_patches as _patch_aiohttp_trust_env  # legacy alias


def _try_agent_toolkit(refs: list[CulturalRef]) -> list[MappedRef] | None:
    """Run Part 2 through the NeMo Agent Toolkit ReAct workflow. On any failure
    (import, config load, agent error), returns None so the caller can fall back
    to the deterministic chain."""
    try:
        import asyncio
        _patch_aiohttp_trust_env()
        from nat.runtime.loader import load_workflow  # type: ignore
        # Importing nat_tools triggers @register_function side effects.
        from src.part2_cultural import nat_tools  # noqa: F401
    except ImportError as exc:
        print(f"[agent_runner] NAT unavailable ({exc}); using deterministic fallback.")
        return None

    async def _run_all() -> list[MappedRef]:
        out: list[MappedRef] = []
        async with load_workflow(str(CONFIG_PATH)) as session:
            for ref in refs:
                prompt = (
                    f'Map the English cultural reference "{ref.term}" (type: {ref.type}) '
                    "to its best Korean cultural equivalent. Follow the dict → retriever → "
                    "web search order. Return JSON only."
                )
                async with session.run(prompt) as runner:
                    result = await runner.result()
                text = result if isinstance(result, str) else getattr(result, "content", str(result))
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    # Best-effort extraction of the JSON object from the agent's free text.
                    start, end = text.find("{"), text.rfind("}")
                    payload = json.loads(text[start:end + 1]) if 0 <= start < end else {}
                payload.setdefault("term", ref.term)
                payload.setdefault("type", ref.type)
                payload.setdefault("ko", ref.term)
                payload.setdefault("source", "web+llm")
                payload.setdefault("retrieved", payload["source"] == "web+llm")
                payload.setdefault("notes", "")
                out.append(MappedRef.model_validate(payload))
        return out

    try:
        return asyncio.run(_run_all())
    except Exception as exc:
        print(f"[agent_runner] NAT runtime error ({type(exc).__name__}: {exc}); "
              "falling back to deterministic chain.")
        return None


def map_refs(refs: list[CulturalRef], *, use_nat: bool | None = None) -> list[MappedRef]:
    """Map cultural references to Korean equivalents.

    `use_nat` controls which path runs:
      - None (default) → read MAP_REFS_USE_NAT env var; default False.
      - True → route through the NeMo Agent Toolkit ReAct workflow (demo path);
        if NAT fails at any stage, fall back to the deterministic chain.
      - False → use the deterministic Python chain (dict → retriever → web+llm)
        for reproducible pipeline output.

    The deterministic chain is the production default because it reliably honors
    the dict → retriever → web ordering and never re-asks Nemotron for terms
    already in the curated map. The NAT path is a genuine agent and occasionally
    skips dict lookups; useful for showcasing the toolkit on select inputs."""
    if use_nat is None:
        use_nat = os.environ.get("MAP_REFS_USE_NAT", "").lower() in {"1", "true", "yes"}
    if use_nat:
        via_toolkit = _try_agent_toolkit(refs)
        if via_toolkit is not None:
            return via_toolkit
    return [_map_one_direct(r) for r in refs]
