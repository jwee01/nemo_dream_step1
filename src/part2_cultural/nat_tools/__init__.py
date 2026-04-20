"""Importing this package side-effect-registers the three cultural-mapper tools
with NAT's function registry so a YAML workflow can reference them by `_type`."""

from src.part2_cultural.nat_tools import dict_lookup_tool  # noqa: F401
from src.part2_cultural.nat_tools import retriever_tool  # noqa: F401
from src.part2_cultural.nat_tools import web_tool  # noqa: F401
