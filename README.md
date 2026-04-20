# Data Pipeline Phase 1 — Sociolinguistic Decomposition + Cultural Substitution

NVIDIA Nemotron Hackathon **Track C: Nemotron for SDG**. Stages 1–2 of a pipeline that converts English SNS datasets into Korean datasets for k‑sovereign LLM training. **Not translation** — rewriting with cultural adaptation.

**Owned here:** Stage 1 (sociolinguistic JSON extraction) + Stage 2 (cultural reference substitution).
**Owned by teammates:** Stage 3 (Korean rewriting), Stage 4 (rule-based marker injection), Stage 5 (automatic evaluation).

---

## What it does

```
English SNS post                                  Stage12Output (per row)
"happy thanksgiving fam,       ──────────►       {
 eating way too much turkey"                       source_text, decomposed, mapped_refs
                                                 }

Stage 1: Sociolinguistic decomposition → extract speech_act, register, emotion,
         cultural_refs[{type, term}], internet_markers, age_group, platform_fit
Stage 2: Cultural reference substitution → each term → {ko, source, retrieved, notes}
         e.g. thanksgiving → 추석 (holiday), venmo → 토스 (service)
```

Stage 2 tries three routes in order: **local dict → embedding retriever → web+LLM fallback**. Fresh web+LLM results are auto-appended back to the dictionary (self-growing cache).

---

## NVIDIA stack

| Where | NVIDIA tool | Model | Role |
|---|---|---|---|
| Stage 1 primary (batch) | **NeMo Data Designer** (`data-designer`) | Nemotron-3-Nano-30B-a3b | Structured dataset generation with Pydantic schema |
| Stage 1 fallback | NIM OpenAI-compat + `nvext.guided_json` (XGrammar) | Nemotron-3-Nano-30B-a3b | Sequential direct calls for debugging / DD failure |
| Stage 2 retriever | **NeMo Retriever** | `nvidia/llama-3.2-nv-embedqa-1b-v2` | Korean-capable embedding similarity over the cultural_map |
| Stage 2 agent (opt-in) | **NeMo Agent Toolkit** (`nvidia-nat`, `nvidia-nat-langchain`) | Nemotron-3-Nano-30B-a3b | `tool_calling_agent` ReAct with 3 custom tools |
| Stage 2 reasoning | NIM OpenAI-compat + guided_json | Nemotron-3-Nano-30B-a3b | Korean-equivalent reasoning with web snippets |

All LLM / embedding traffic goes through `https://integrate.api.nvidia.com/v1` (build.nvidia.com, OpenAI-compatible).

---

## Repo layout

```
├── src/
│   ├── schemas.py                         # Pydantic — single source of truth for row shape
│   ├── io_loader.py                       # JSONL / JSON / HuggingFace loader
│   ├── proxy_patch.py                     # corporate proxy monkey-patch for aiohttp/httpx
│   ├── part1_decompose/
│   │   ├── prompts.py                     # system prompt + few-shot examples
│   │   ├── data_designer_runner.py        # Stage 1: DD primary + sequential fallback
│   │   └── nim_guided_json.py             # direct-NIM call + output normalizer
│   └── part2_cultural/
│       ├── agent_runner.py                # Stage 2: deterministic chain (default) + NAT (opt-in)
│       ├── index_builder.py               # builds cultural_map embedding index
│       ├── tools/                         # sync Python tools (dict, retriever, web)
│       └── nat_tools/                     # NAT `@register_function` wrappers
├── configs/
│   ├── cultural_map.json                  # seed dict + auto-growing `retrieved` section
│   ├── data_designer.yaml                 # DD config (batch Stage 1)
│   ├── cultural_agent.yaml                # NAT workflow (opt-in Stage 2)
│   └── retriever_index.npz                # 29×2048 float32 embeddings (built on demand)
├── scripts/
│   ├── run_pipeline.py                    # end-to-end CLI
│   └── slurm/                             # sbatch wrappers for the cluster
├── tests/
│   ├── sample_inputs.py                   # 20 hardcoded samples
│   ├── smoke_test.py                      # end-to-end smoke (5+1)
│   ├── stability_test.py                  # N trials × M samples, label variance
│   └── verify_nvidia_paths.py             # confirm DD & NAT primary paths actually fire
└── sample_data.jsonl                      # example input for run_pipeline
```

---

## Quickstart

### 0. One-time setup

```bash
cp .env.example .env
# Edit .env — fill in NVIDIA_API_KEY (from build.nvidia.com) and TAVILY_API_KEY
```

Required env vars (`.env.example`):
- `NVIDIA_API_KEY` — from [build.nvidia.com](https://build.nvidia.com/)
- `TAVILY_API_KEY` — from [tavily.com](https://tavily.com) (Stage 2 web fallback)
- `NVIDIA_API_BASE` — default `https://integrate.api.nvidia.com/v1`
- `NEMOTRON_MODEL` — default `nvidia/nemotron-3-nano-30b-a3b`
- `RETRIEVER_MODEL` — default `nvidia/llama-3.2-nv-embedqa-1b-v2`

### 1. Install (one time)

```bash
sbatch scripts/slurm/install.sbatch         # uv venv + base deps
sbatch scripts/slurm/install_extras.sbatch  # adds nvidia-nat[langchain] for NAT agent
```

These create `.venv/` (Python 3.11) and install everything.

### 2. Build the retriever embedding index (one time, re-run when `cultural_map.json` changes significantly)

```bash
sbatch scripts/slurm/build_index.sbatch
```

Produces `configs/retriever_index.npz`.

### 3. Run the pipeline

```bash
# Full dataset
sbatch scripts/slurm/run_pipeline.sbatch --input sample_data.jsonl --output stage12.jsonl

# With a row cap
sbatch scripts/slurm/run_pipeline.sbatch --input sample_data.jsonl --output stage12.jsonl --limit 5

# HuggingFace dataset
sbatch scripts/slurm/run_pipeline.sbatch --input hf:Anthropic/hh-rlhf:train --output stage12.jsonl --limit 100
```

Watch `logs/pipeline-<jobid>.out` / `.err` for progress. Final output: `stage12.jsonl`.

### 4. Verification / diagnostics

```bash
sbatch scripts/slurm/smoke.sbatch        # 5-sample end-to-end smoke
sbatch scripts/slurm/stability.sbatch --trials 3 --samples 8   # label-variance test
sbatch scripts/slurm/verify_nvidia.sbatch  # confirm DD + NAT primary paths actually execute
```

---

## Output schema

Each line of `stage12.jsonl` is one `Stage12Output`:

```jsonc
{
  "id": "d05",
  "source_text": "bro literally got his first paycheck ... at Google and already splurged on a new iphone lmao",
  "decomposed": {
    "speech_act": "statement",
    "register": "casual",
    "emotion": {"type": "surprise", "intensity": 3},
    "cultural_refs": [
      {"type": "brand", "term": "google"},
      {"type": "brand", "term": "iphone"}
    ],
    "internet_markers": {"laughter": "lmao", "emphasis": [], "sarcasm_marker": false},
    "estimated_age_group": "20s",
    "platform_fit": ["twitter", "reddit"]
  },
  "mapped_refs": [
    {"term": "google",  "ko": "네이버",   "type": "brand",    "source": "dict",    "retrieved": false, "notes": ""},
    {"term": "iphone",  "ko": "아이폰",   "type": "brand",    "source": "web+llm", "retrieved": true,  "notes": "..."}
  ]
}
```

Interpreting `mapped_refs[i].source`:
- `"dict"` — curated seed or cached retrieval (highest trust)
- `"retriever"` — embedding-similarity fuzzy match (medium trust)
- `"web+llm"` — web-grounded Nemotron inference (review recommended)

See [src/schemas.py](src/schemas.py) for the full Pydantic definition.

---

## Flow under the hood

```
run_pipeline.sbatch
 └─ run_pipeline.py
     ├─ io_loader.load(source, limit) → [{id, text}, ...]
     ├─ part1_decompose.data_designer_runner.run_rows(rows)
     │    ├─ try: NeMo Data Designer batch (Nemotron via NIM)
     │    └─ fallback: sequential nim_guided_json.decompose per row
     │    → writes <output>.stage1.jsonl as intermediate
     └─ for each row: part2_cultural.agent_runner.map_refs(cultural_refs)
          └─ deterministic chain:
               dict_lookup(term)                 → "dict"       (instant)
               retriever_search(term, 0.65)      → "retriever"  (NeMo Retriever embedding)
               web_search + Nemotron reasoning   → "web+llm"    (+ auto-append to cultural_map.json)
     → writes <output> (stage12.jsonl) — one Stage12Output per line
```

`MAP_REFS_USE_NAT=1` env var switches Stage 2 to the NeMo Agent Toolkit ReAct workflow instead of the deterministic chain.

---

## Cluster / Slurm notes

- **Login node is shared** — never run long jobs inline. All multi-second work goes through `sbatch`.
- System Python is 3.9; we use `uv venv` to get 3.11 (NVIDIA packages require >=3.10).
- The compute nodes sit behind an internal HTTPS proxy. `src/proxy_patch.py` monkey-patches `aiohttp` and `httpx`-with-custom-transport so NAT and Data Designer can reach `build.nvidia.com`. Safe no-op if no proxy is set.
- All sbatch scripts target partition `cpu` (API-only workloads don't need a GPU).

---

## Known limitations

- **NIM `guided_json` enum enforcement is soft** — the managed endpoint often returns values outside the declared Literal set. `_normalize()` in `nim_guided_json.py` guards with alias maps and range clamps.
- **`register` field in `Decomposed` shadows a Pydantic `BaseModel` attribute** → harmless `UserWarning` at import. Leaving as-is; renaming would change the downstream contract.
- **Data Designer 0.5.7 stats-phase bug** (`ValueError: truth value of an array ... ambiguous`) fires AFTER records are generated and written. We catch it and load the parquet directly.
- **NAT agent planning is probabilistic** — it occasionally chooses `web+llm` for terms already in the dict. This is why the default path is the deterministic chain; NAT is opt-in for demos.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` in sbatch | Ensure script does `PYTHONPATH="$PWD" python ...`. Already the case for the shipped sbatch wrappers. |
| `ClientConnectorError` / `ConnectError: All connection attempts failed` on compute node | Missing proxy patch. Make sure `src.proxy_patch.apply_proxy_patches()` runs before any NAT / DD call. |
| DD log shows `timed out while running health checks` | Already mitigated: `ModelConfig(skip_health_check=True)`. |
| DD fails with `No such file or directory: 'parquet-files'` | DD generation itself failed (look at warnings above in stderr). Usually proxy or model issue. |
| Smoke test `Expected all dict terms to hit dict` | Someone corrupted `cultural_map.json` or `retriever_search` threshold is misconfigured. |

Logs: `logs/<jobname>-<jobid>.out` / `.err`.
