"""NAT-compatible wrapper for the dictionary lookup tool. Delegates to the existing
sync implementation in `src.part2_cultural.tools.dict_lookup`.

Note: do NOT use `from __future__ import annotations` in this file — NAT introspects
signatures via `inspect.signature(...).parameters[i].annotation` and needs the real
classes, not stringified forward references."""

import json

from src.part2_cultural.tools import dict_lookup as _impl

try:
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig

    class DictLookupConfig(FunctionBaseConfig, name="cultural_dict_lookup"):
        pass

    @register_function(config_type=DictLookupConfig)
    async def dict_lookup_tool(_config: DictLookupConfig, _builder: Builder):  # type: ignore[no-redef]
        async def _lookup(term: str) -> str:
            hit = _impl.lookup(term)
            return json.dumps(hit or {}, ensure_ascii=False)

        yield FunctionInfo.from_fn(
            _lookup,
            description=(
                "Exact/normalized lookup against the curated Korean cultural_map. "
                "Input: an English cultural reference term (string). "
                "Returns JSON with fields {en, ko, type, notes, source} if known, "
                "or '{}' on miss. Call first for any cultural reference."
            ),
        )
except ImportError:
    pass
