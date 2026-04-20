"""Stage 1 batch runner. Primary path uses NeMo Data Designer
(`DataDesignerConfigBuilder` + `DataDesigner.create`); falls back to sequential
direct NIM calls if the Data Designer API shape differs from expected or the
backend errors out.

Both paths produce the same shape: for each input row `{id, text}`, emit
`{id, text, decomposed: <Decomposed dict>}`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from src.part1_decompose.nim_guided_json import decompose, _normalize
from src.part1_decompose.prompts import SYSTEM_PROMPT
from src.proxy_patch import apply_proxy_patches
from src.schemas import Decomposed


def _try_data_designer(input_rows: list[dict], artifact_path: Path) -> Optional[list[dict]]:
    """Run via NeMo Data Designer. Returns list of stage1 rows, or None on failure."""
    try:
        apply_proxy_patches()
        import pandas as pd
        from data_designer.interface import DataDesigner
        from data_designer.config.config_builder import DataDesignerConfigBuilder
        from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
        from data_designer.config.seed_source import LocalFileSeedSource
    except ImportError as exc:
        print(f"[data_designer] import failed ({exc}); using sequential NIM fallback.")
        return None

    try:
        nim_provider = ModelProvider(
            name="nvidia-nim",
            endpoint=os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
            provider_type="openai",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        model_config = ModelConfig(
            alias="nemotron-nano",
            model=os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
            provider="nvidia-nim",
            skip_health_check=True,  # health check hits timeouts; rely on actual calls
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.2,
                max_parallel_requests=2,
                timeout=180,
            ),
        )

        artifact_path.mkdir(parents=True, exist_ok=True)
        seed_path = artifact_path / "seed.jsonl"
        seed_df = pd.DataFrame({"source_text": [r["text"] for r in input_rows]})
        seed_df.to_json(seed_path, orient="records", lines=True, force_ascii=False)

        builder = DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
        # `with_seed_dataset` auto-adds the seed columns; only add the LLM-structured column.
        builder.add_column(
            name="decomposed",
            column_type="llm-structured",
            prompt=SYSTEM_PROMPT + '\n\nAnnotate this post:\n"""{{ source_text }}"""',
            model_alias="nemotron-nano",
            output_format=Decomposed,
        )

        designer = DataDesigner(
            artifact_path=str(artifact_path),
            model_providers=[nim_provider],
        )
        df = None
        try:
            result = designer.create(
                config_builder=builder,
                num_records=len(seed_df),
                dataset_name="stage1_decomposed",
            )
            df = result.load_dataset() if hasattr(result, "load_dataset") else result
        except Exception as dd_exc:
            # DD 0.5.7 has a stats-phase bug ("truth value of an array ... is ambiguous")
            # that triggers AFTER all records have been successfully generated and
            # written to parquet. Recover by reading the latest parquet batch directly.
            print(f"[data_designer] create() post-gen error ({type(dd_exc).__name__}); "
                  "attempting to recover from parquet artifacts.")
            import glob
            pattern = str(artifact_path / "stage1_decomposed*" / "parquet-files" / "*.parquet")
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                raise
            df = pd.concat([pd.read_parquet(p) for p in candidates[-len(seed_df):]],
                           ignore_index=True)

        def _to_jsonable(obj):
            """Recursively convert numpy arrays / scalars back to plain Python lists/values."""
            try:
                import numpy as np
            except ImportError:  # pragma: no cover
                np = None  # type: ignore[assignment]
            if np is not None and isinstance(obj, np.ndarray):
                return [_to_jsonable(x) for x in obj.tolist()]
            if np is not None and isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            return obj

        out: list[dict] = []
        for original, (_, row) in zip(input_rows, df.iterrows()):
            payload = row["decomposed"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = _to_jsonable(payload)
            payload = _normalize(payload, original["text"])
            out.append({
                "id": original.get("id"),
                "text": original["text"],
                "decomposed": Decomposed.model_validate(payload).model_dump(),
            })
        print(f"[data_designer] produced {len(out)} rows via NeMo Data Designer ✓")
        return out
    except Exception as exc:
        print(f"[data_designer] runtime error ({type(exc).__name__}: {exc}); "
              "falling back to sequential NIM.")
        return None


def _sequential(input_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in tqdm(input_rows, desc="stage1-seq"):
        dec = decompose(r["text"])
        out.append({
            "id": r.get("id"),
            "text": r["text"],
            "decomposed": dec.model_dump(),
        })
    return out


def run_rows(input_rows: list[dict], config_path: Optional[Path] = None) -> list[dict]:
    """High-level entrypoint: accepts a list of `{id, text}` dicts, returns a list of
    stage1 rows `{id, text, decomposed}`. Tries Data Designer first, then falls back."""
    load_dotenv()
    artifact_path = Path(__file__).resolve().parents[2] / "artifacts" / "data_designer"
    via_dd = _try_data_designer(input_rows, artifact_path)
    return via_dd if via_dd is not None else _sequential(input_rows)


def run_batch(input_path: Path, output_path: Path, config_path: Path) -> None:
    """File-in / file-out wrapper kept for compatibility with older callers."""
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    rows = [{"id": r.get("id"), "text": r.get("text") or r.get("source_text", "")} for r in rows]
    stage1_rows = run_rows(rows, config_path)
    with output_path.open("w") as f:
        for r in stage1_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[data_designer_runner] wrote {len(stage1_rows)} rows → {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "data_designer.yaml",
    )
    args = ap.parse_args()
    run_batch(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
