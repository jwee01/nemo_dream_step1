"""NAT-compatible wrapper for the NeMo Retriever (embedding similarity) tool."""

import json

from src.part2_cultural.tools import retriever_search as _impl

try:
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig

    class RetrieverSearchConfig(FunctionBaseConfig, name="cultural_retriever_search"):
        pass

    @register_function(config_type=RetrieverSearchConfig)
    async def retriever_search_tool(_config: RetrieverSearchConfig, _builder: Builder):  # type: ignore[no-redef]
        async def _search(term: str) -> str:
            hits = _impl.search(term, top_k=3, threshold=0.65)
            return json.dumps(hits, ensure_ascii=False)

        yield FunctionInfo.from_fn(
            _search,
            description=(
                "NeMo Retriever embedding similarity search over the cultural_map. "
                "Input: an English term (string). Returns a JSON list of up to 3 "
                "candidate entries with cosine scores >= 0.65. Use when dict_lookup "
                "misses."
            ),
        )
except ImportError:
    pass
