"""Full Stage 1+2 pipeline. Reads a source (JSONL path or `hf:<dataset>[:<split>]`)
and writes a JSONL with one `Stage12Output` per row. Always submit via sbatch for
non-trivial inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.io_loader import load, write_jsonl
from src.part1_decompose.data_designer_runner import run_rows as run_stage1_rows
from src.part2_cultural.agent_runner import map_refs
from src.schemas import CulturalRef, Decomposed, Stage12Output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="JSONL path, JSON path, or `hf:<dataset>[:<split>]`")
    ap.add_argument("--output", type=Path, required=True, help="JSONL with Stage12Output")
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap")
    ap.add_argument(
        "--stage1-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "data_designer.yaml",
    )
    args = ap.parse_args()

    input_rows = load(args.input, limit=args.limit)
    print(f"[run_pipeline] loaded {len(input_rows)} rows from {args.input}")

    stage1_out = args.output.with_suffix(".stage1.jsonl")
    stage1_rows = run_stage1_rows(input_rows, args.stage1_config)
    write_jsonl(stage1_rows, stage1_out)

    with args.output.open("w") as fout:
        for row in tqdm(stage1_rows, desc="stage2"):
            decomposed = Decomposed.model_validate(row["decomposed"])
            refs = [CulturalRef.model_validate(r) for r in decomposed.cultural_refs]
            mapped = map_refs(refs)
            out_row = Stage12Output(
                source_text=decomposed.source_text,
                decomposed=decomposed,
                mapped_refs=mapped,
            )
            fout.write(json.dumps(
                {"id": row.get("id"), **out_row.model_dump()},
                ensure_ascii=False,
            ) + "\n")

    print(f"[run_pipeline] wrote {args.output}")


if __name__ == "__main__":
    main()
