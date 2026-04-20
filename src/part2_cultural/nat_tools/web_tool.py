"""NAT-compatible wrapper for the Tavily-backed web search fallback tool."""

import json

from src.part2_cultural.tools import web_search as _impl

try:
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig

    class WebSearchConfig(FunctionBaseConfig, name="cultural_web_search"):
        pass

    @register_function(config_type=WebSearchConfig)
    async def web_search_tool(_config: WebSearchConfig, _builder: Builder):  # type: ignore[no-redef]
        async def _search(query: str) -> str:
            results = _impl.search(query, max_results=3)
            return json.dumps(results, ensure_ascii=False)

        yield FunctionInfo.from_fn(
            _search,
            description=(
                "Web search (Tavily) fallback for cultural references not in dict or "
                "retriever. Input: an English term (string). Returns a JSON list of "
                "title/content/url snippets. Use only after dict_lookup and "
                "retriever_search both miss."
            ),
        )
except ImportError:
    pass
