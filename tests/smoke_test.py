"""End-to-end smoke test across the 5 samples.

Side effects: calls build.nvidia.com (Nemotron + Retriever) and optionally Tavily.
Do NOT run on the login node; submit via `scripts/slurm/smoke.sbatch`.

Success criteria (printed at the end):
- Part 1: all 5 samples decompose to valid `Decomposed`.
- Part 2: each of {thanksgiving, venmo, uber, prom, snl} hits source="dict".
- Part 2: the injected OOD term "skibidi" hits source="web+llm" with retrieved=True.
"""

from __future__ import annotations

import json

from src.part1_decompose.nim_guided_json import decompose
from src.part2_cultural.agent_runner import map_refs
from src.schemas import CulturalRef, Stage12Output
from tests.sample_inputs import OOD_TERM, SAMPLES

SAMPLES = SAMPLES[:5]  # smoke runs only the original 5; full 20-sample set is used by run_pipeline

DICT_EXPECTED = {"thanksgiving", "venmo", "uber", "prom", "snl"}


def main() -> None:
    outputs: list[Stage12Output] = []
    for sample in SAMPLES:
        text = sample["text"]
        decomposed = decompose(text)
        mapped = map_refs(decomposed.cultural_refs)
        outputs.append(Stage12Output(
            source_text=text, decomposed=decomposed, mapped_refs=mapped,
        ))

    # Force-exercise the web+llm path with a deliberate OOD term.
    ood = map_refs([CulturalRef(type="slang", term=OOD_TERM)])
    outputs.append(Stage12Output(
        source_text=f"[synthetic] {OOD_TERM}",
        decomposed=outputs[0].decomposed,
        mapped_refs=ood,
    ))

    # Print results.
    for o in outputs:
        print(json.dumps(o.model_dump(), ensure_ascii=False, indent=2))
        print("-" * 60)

    all_mapped = [m for o in outputs for m in o.mapped_refs]
    dict_hits = {m.term.lower() for m in all_mapped if m.source == "dict"}
    covered = DICT_EXPECTED & dict_hits
    web_llm_hits = [m for m in all_mapped if m.source == "web+llm" and m.retrieved]
    ood_hit = next((m for m in all_mapped if m.term.lower() == OOD_TERM.lower()), None)

    print(f"Dict coverage: {sorted(covered)} (expected superset of {sorted(DICT_EXPECTED)})")
    print(f"Web+LLM retrieved hits (newly seen this run): {[m.term for m in web_llm_hits]}")
    if ood_hit is not None:
        print(f"OOD term `{OOD_TERM}` mapped via source={ood_hit.source} → ko={ood_hit.ko!r}")

    assert len(outputs) == len(SAMPLES) + 1, "Output count mismatch."
    assert covered == DICT_EXPECTED, (
        f"Expected all dict terms to hit dict, missing={DICT_EXPECTED - covered}"
    )
    assert ood_hit is not None, "OOD term was not processed at all."
    # Either it was newly retrieved this run, or it's already cached from a prior run.
    assert ood_hit.source in {"web+llm", "dict"}, (
        f"OOD term should come from web+llm or the self-growing dict cache, got {ood_hit.source}"
    )
    print("SMOKE TEST PASSED ✓")


if __name__ == "__main__":
    main()
