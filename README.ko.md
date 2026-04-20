# 데이터 파이프라인 Phase 1 — 사회언어학적 분해 + 문화적 요소 치환

Also available in: **[English](README.md)**

NVIDIA Nemotron 해커톤 **Track C: Nemotron for SDG** 출품작의 Stage 1-2. 영어 SNS 데이터셋을 k-소버린(한국 문화 이해도 높은) LLM 학습용 한국어 데이터셋으로 변환하는 파이프라인. **단순 번역이 아니라 문화적 적응을 포함한 재작성**.

**이 저장소가 담당:** Stage 1 (사회언어학적 JSON 추출) + Stage 2 (문화 레퍼런스 치환).
**팀원 담당:** Stage 3 (한국어 재작성), Stage 4 (룰 기반 마커 주입), Stage 5 (자동 평가).

---

## 무엇을 하는가

```
영어 SNS 포스트                                     Stage12Output (행당)
"happy thanksgiving fam,       ──────────►        {
 eating way too much turkey"                        source_text, decomposed, mapped_refs
                                                  }

Stage 1: 사회언어학적 분해 → speech_act(화행), register(공식성), emotion(감정),
         cultural_refs[{type, term}](문화참조), internet_markers(인터넷 마커),
         estimated_age_group(연령), platform_fit(플랫폼) 추출
Stage 2: 문화 레퍼런스 치환 → 각 term에 한국 등가물 매핑
         예: thanksgiving → 추석 (holiday), venmo → 토스 (service)
```

Stage 2는 세 경로를 순서대로 시도: **로컬 사전 → 임베딩 기반 유사도 검색 → 웹 검색+LLM 추론**. 웹+LLM으로 새로 매핑된 결과는 **자동으로 사전에 누적** (self-growing cache).

---

## NVIDIA 스택

| 단계 | NVIDIA 도구 | 모델 | 역할 |
|---|---|---|---|
| Stage 1 primary (배치) | **NeMo Data Designer** (`data-designer`) | Nemotron-3-Nano-30B-a3b | Pydantic 스키마 기반 구조화 데이터셋 생성 |
| Stage 1 fallback | NIM OpenAI 호환 + `nvext.guided_json` (XGrammar) | Nemotron-3-Nano-30B-a3b | 순차 직접 호출 — 디버깅 / DD 실패 시 |
| Stage 2 retriever | **NeMo Retriever** | `nvidia/llama-3.2-nv-embedqa-1b-v2` | 한국어 지원 임베딩으로 cultural_map 유사도 검색 |
| Stage 2 agent (opt-in) | **NeMo Agent Toolkit** (`nvidia-nat`, `nvidia-nat-langchain`) | Nemotron-3-Nano-30B-a3b | 3개 커스텀 툴 있는 `tool_calling_agent` ReAct |
| Stage 2 reasoning | NIM OpenAI 호환 + guided_json | Nemotron-3-Nano-30B-a3b | 웹 snippet 근거로 한국 등가물 추론 |

모든 LLM / 임베딩 호출은 `https://integrate.api.nvidia.com/v1` (build.nvidia.com, OpenAI 호환) 경유.

---

## 디렉토리 구조

```
├── src/
│   ├── schemas.py                         # Pydantic — 데이터 shape의 단일 출처
│   ├── io_loader.py                       # JSONL / JSON / HuggingFace 로더
│   ├── proxy_patch.py                     # 사내 프록시용 aiohttp/httpx monkey-patch
│   ├── part1_decompose/
│   │   ├── prompts.py                     # 시스템 프롬프트 + few-shot 예시
│   │   ├── data_designer_runner.py        # Stage 1: DD primary + 순차 fallback
│   │   └── nim_guided_json.py             # NIM 직접 호출 + 출력 normalizer
│   └── part2_cultural/
│       ├── agent_runner.py                # Stage 2: 결정론적 체인 (기본) + NAT (opt-in)
│       ├── index_builder.py               # cultural_map 임베딩 인덱스 빌더
│       ├── tools/                         # 동기 Python 툴 (dict, retriever, web)
│       └── nat_tools/                     # NAT `@register_function` 래퍼
├── configs/
│   ├── cultural_map.json                  # 시드 사전 + 자동 성장 `retrieved` 섹션
│   ├── data_designer.yaml                 # DD 설정 (배치 Stage 1)
│   ├── cultural_agent.yaml                # NAT 워크플로 (opt-in Stage 2)
│   └── retriever_index.npz                # 29×2048 float32 임베딩 (요청 시 빌드)
├── scripts/
│   ├── run_pipeline.py                    # end-to-end CLI
│   └── slurm/                             # 클러스터용 sbatch 래퍼
├── tests/
│   ├── sample_inputs.py                   # 하드코딩 샘플 20개
│   ├── smoke_test.py                      # end-to-end smoke (5+1)
│   ├── stability_test.py                  # N trial × M sample, 라벨 분산
│   └── verify_nvidia_paths.py             # DD · NAT primary path 실발동 확인
└── sample_data.jsonl                      # run_pipeline 예시 입력
```

---

## 빠르게 시작하기

### 0. 초기 셋업 (최초 1회)

```bash
cp .env.example .env
# .env 편집 — NVIDIA_API_KEY 와 TAVILY_API_KEY 입력
```

필요 환경 변수 (`.env.example` 참고):
- `NVIDIA_API_KEY` — [build.nvidia.com](https://build.nvidia.com/) 에서 발급
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com) 에서 발급 (Stage 2 웹 검색)
- `NVIDIA_API_BASE` — 기본값 `https://integrate.api.nvidia.com/v1`
- `NEMOTRON_MODEL` — 기본값 `nvidia/nemotron-3-nano-30b-a3b`
- `RETRIEVER_MODEL` — 기본값 `nvidia/llama-3.2-nv-embedqa-1b-v2`

### 1. 설치 (최초 1회)

```bash
sbatch scripts/slurm/install.sbatch         # uv venv + 기본 의존성
sbatch scripts/slurm/install_extras.sbatch  # NAT 에이전트용 nvidia-nat[langchain] 추가
```

Python 3.11 `.venv/` 생성하고 모든 의존성 설치.

### 2. Retriever 임베딩 인덱스 빌드 (사전 크게 바뀔 때마다 재실행)

```bash
sbatch scripts/slurm/build_index.sbatch
```

`configs/retriever_index.npz` 생성.

### 3. 파이프라인 실행

```bash
# 전체 데이터셋
sbatch scripts/slurm/run_pipeline.sbatch --input sample_data.jsonl --output stage12.jsonl

# 행 수 제한
sbatch scripts/slurm/run_pipeline.sbatch --input sample_data.jsonl --output stage12.jsonl --limit 5

# HuggingFace 데이터셋
sbatch scripts/slurm/run_pipeline.sbatch --input hf:Anthropic/hh-rlhf:train --output stage12.jsonl --limit 100
```

진행 상황은 `logs/pipeline-<jobid>.out` / `.err`. 최종 산출물: `stage12.jsonl`.

### 4. 검증 / 진단

```bash
sbatch scripts/slurm/smoke.sbatch                                # 5샘플 end-to-end smoke
sbatch scripts/slurm/stability.sbatch --trials 3 --samples 8     # 라벨 안정성
sbatch scripts/slurm/verify_nvidia.sbatch                        # DD + NAT 실제 발동 확인
```

---

## 출력 스키마

`stage12.jsonl` 각 줄은 `Stage12Output` 하나:

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

`mapped_refs[i].source` 해석:
- `"dict"` — 큐레이션 시드 또는 자동 누적된 캐시 (신뢰도 최고)
- `"retriever"` — 임베딩 유사도 매칭 (중간 신뢰도)
- `"web+llm"` — 웹 기반 Nemotron 추론 (수동 검수 권장)

전체 Pydantic 정의: [src/schemas.py](src/schemas.py).

---

## 내부 흐름

```
run_pipeline.sbatch
 └─ run_pipeline.py
     ├─ io_loader.load(source, limit) → [{id, text}, ...]
     ├─ part1_decompose.data_designer_runner.run_rows(rows)
     │    ├─ 시도: NeMo Data Designer 배치 (NIM 통해 Nemotron)
     │    └─ fallback: 행마다 nim_guided_json.decompose 순차 호출
     │    → 중간 파일 <output>.stage1.jsonl 저장
     └─ 각 행마다: part2_cultural.agent_runner.map_refs(cultural_refs)
          └─ 결정론적 체인:
               dict_lookup(term)                 → "dict"       (즉시)
               retriever_search(term, 0.65)      → "retriever"  (NeMo Retriever 임베딩)
               web_search + Nemotron 추론        → "web+llm"    (+ cultural_map.json 자동 추가)
     → 최종 <output> (stage12.jsonl) — 행당 하나의 Stage12Output
```

`MAP_REFS_USE_NAT=1` 환경변수 설정 시 Stage 2가 결정론 체인 대신 NeMo Agent Toolkit ReAct 워크플로로 전환.

---

## 클러스터 / Slurm 주의사항

- **로그인 노드는 공유** — 긴 작업을 인라인 실행하지 말 것. 모든 수초 이상 작업은 `sbatch` 경유.
- 시스템 Python은 3.9 — NVIDIA 패키지들이 3.10+ 요구하므로 `uv venv`로 3.11 사용.
- 컴퓨트 노드는 사내 HTTPS 프록시 뒤에 있음. `src/proxy_patch.py`가 `aiohttp`와 custom-transport `httpx`에 monkey-patch를 적용해 NAT와 Data Designer가 `build.nvidia.com`에 도달 가능하게 함. 프록시 없는 환경에서는 자동으로 no-op.
- 모든 sbatch는 `cpu` 파티션 지정 (API 호출 기반 워크로드는 GPU 불필요).

---

## 알려진 한계

- **NIM `guided_json` enum 제약이 소프트** — 관리형 엔드포인트가 Literal set 밖 값을 종종 반환. `nim_guided_json.py`의 `_normalize()`가 alias 맵과 range clamp로 가드.
- **`Decomposed`의 `register` 필드가 Pydantic `BaseModel` 속성을 가림** → import 시 무해한 `UserWarning`. 하위 contract 유지를 위해 그대로 둠.
- **Data Designer 0.5.7의 stats 단계 버그** (`ValueError: truth value of an array ... ambiguous`)가 레코드 생성·저장 후에 터짐. 우리는 이 에러를 catch해 parquet 파일을 직접 읽어옴.
- **NAT 에이전트 플래닝은 확률적** — 사전에 이미 있는 용어를 가끔 `web+llm`으로 처리. 그래서 기본 경로는 결정론 체인, NAT는 데모용 opt-in.

---

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| sbatch에서 `ModuleNotFoundError: No module named 'src'` | 스크립트가 `PYTHONPATH="$PWD" python ...` 호출하는지 확인. 제공된 sbatch 래퍼들은 이미 설정됨. |
| 컴퓨트 노드에서 `ClientConnectorError` / `ConnectError: All connection attempts failed` | 프록시 패치 누락. NAT / DD 호출 전에 `src.proxy_patch.apply_proxy_patches()` 실행되는지 확인. |
| DD 로그에 `timed out while running health checks` | 이미 완화됨: `ModelConfig(skip_health_check=True)`. |
| DD 에러 `No such file or directory: 'parquet-files'` | DD 생성 자체 실패 (stderr 상단 warning 참고). 대개 프록시 또는 모델 이슈. |
| smoke 실패 `Expected all dict terms to hit dict` | `cultural_map.json`이 훼손됐거나 `retriever_search` threshold 오설정. |

로그 위치: `logs/<jobname>-<jobid>.out` / `.err`.

---

## 파일별 핵심 요약 (코드 읽는 순서 추천)

읽는 순서:
1. [src/schemas.py](src/schemas.py) — 데이터 shape 전부
2. [scripts/run_pipeline.py](scripts/run_pipeline.py) — end-to-end 오케스트레이션
3. [src/part1_decompose/data_designer_runner.py](src/part1_decompose/data_designer_runner.py) — Stage 1
4. [src/part1_decompose/nim_guided_json.py](src/part1_decompose/nim_guided_json.py) — NIM 직접 호출 + normalizer
5. [src/part1_decompose/prompts.py](src/part1_decompose/prompts.py) — 프롬프트 + few-shot
6. [src/part2_cultural/agent_runner.py](src/part2_cultural/agent_runner.py) — Stage 2
7. [src/part2_cultural/tools/](src/part2_cultural/tools/) — dict / retriever / web 3개 툴
8. [src/part2_cultural/nat_tools/](src/part2_cultural/nat_tools/) — NAT 래퍼 (opt-in)
9. [configs/cultural_map.json](configs/cultural_map.json) — 시드 사전 + 자동 누적 캐시
10. [configs/cultural_agent.yaml](configs/cultural_agent.yaml), [configs/data_designer.yaml](configs/data_designer.yaml) — NVIDIA 툴 설정
