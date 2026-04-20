"""Verify that the two NVIDIA-primary paths actually fire (vs falling back):
  - Stage 1: NeMo Data Designer (via run_rows)
  - Stage 2: NeMo Agent Toolkit ReAct agent (via agent_runner._try_agent_toolkit)

Both paths have deterministic fallbacks, so the smoke test alone can't tell us
which code path ran. This script runs them in a way that surfaces the primary-path
log message ("[data_designer] produced ..." / "[agent_runner] NAT ...").

Run via sbatch: scripts/slurm/verify_nvidia.sbatch"""

from __future__ import annotations

from src.part1_decompose.data_designer_runner import run_rows
from src.part2_cultural.agent_runner import _try_agent_toolkit, map_refs
from src.schemas import CulturalRef
from tests.sample_inputs import SAMPLES


def main() -> None:
    print("=" * 60)
    print("STAGE 1 — NeMo Data Designer primary path")
    print("=" * 60)
    stage1_out = run_rows(SAMPLES[:3])
    print(f"  got {len(stage1_out)} rows")
    print(f"  first row decomposed.speech_act = {stage1_out[0]['decomposed']['speech_act']!r}")

    print()
    print("=" * 60)
    print("STAGE 2 — NeMo Agent Toolkit ReAct primary path")
    print("=" * 60)
    refs = [
        CulturalRef(type="holiday", term="thanksgiving"),
        CulturalRef(type="service", term="venmo"),
        CulturalRef(type="slang", term="skibidi"),
    ]
    result = _try_agent_toolkit(refs)
    if result is None:
        print("  [FAIL] NAT primary path returned None (fallback would trigger).")
    else:
        print(f"  [OK] NAT produced {len(result)} mapped refs.")
        for r in result:
            print(f"    - {r.term} -> {r.ko} (source={r.source}, retrieved={r.retrieved})")


if __name__ == "__main__":
    main()
