"""Run Part 1 decomposition N times on the same samples and report the variance of
key labels (speech_act, register, emotion.type, estimated_age_group) per sample.

This tells us how much LLM nondeterminism affects downstream reliability at
temperature 0.2. Run via sbatch because it makes N*|samples| API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from src.part1_decompose.nim_guided_json import decompose
from tests.sample_inputs import SAMPLES


def _mode_and_agreement(values: list[str]) -> tuple[str, float]:
    if not values:
        return ("", 0.0)
    c = Counter(values)
    most, n = c.most_common(1)[0]
    return (most, n / len(values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--samples", type=int, default=len(SAMPLES))
    args = ap.parse_args()

    subset = SAMPLES[:args.samples]
    results: dict[str, dict[str, list[str]]] = {
        s["id"]: {"speech_act": [], "register": [], "emotion": [], "age_group": [],
                  "ref_terms": []} for s in subset
    }

    for trial in range(args.trials):
        print(f"--- trial {trial + 1}/{args.trials} ---")
        for s in subset:
            dec = decompose(s["text"])
            r = results[s["id"]]
            r["speech_act"].append(dec.speech_act)
            r["register"].append(dec.register)
            r["emotion"].append(dec.emotion.type)
            r["age_group"].append(dec.estimated_age_group)
            r["ref_terms"].append(",".join(sorted(cr.term for cr in dec.cultural_refs)))

    # Report
    rows = []
    for sid, r in results.items():
        row = {"id": sid}
        for field in ("speech_act", "register", "emotion", "age_group", "ref_terms"):
            mode, agree = _mode_and_agreement(r[field])
            row[f"{field}_mode"] = mode
            row[f"{field}_agreement"] = round(agree, 3)
        rows.append(row)

    print(json.dumps(rows, ensure_ascii=False, indent=2))

    # Aggregate agreement per field.
    agg = {}
    for field in ("speech_act", "register", "emotion", "age_group", "ref_terms"):
        agg[field] = round(
            sum(row[f"{field}_agreement"] for row in rows) / len(rows), 3
        )
    print("\n=== mean label agreement across samples ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
