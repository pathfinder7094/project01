<<<<<<< HEAD
# Local WikiLLM v4 — Markdown-first + UUID + OpenAI/Ollama + Docling

프로젝트의 목표는 **모델을 학습시키는 것보다 먼저, raw → distill → wiki 파이프라인을 안정적으로 완성하는 것**이다.

- `app/` = 실행 엔진
- `config/` = 실행 규칙
- `llmwiki/` = 독립 Obsidian Vault + canonical knowledge
- `llmwiki/wiki/*.md` = source of truth
- Obsidian = `llmwiki/`를 여는 사용자 인터페이스
- QMD = 선택적 외부 Markdown 검색 엔진
- 별도 KDB = v4에서 사용하지 않음

## 구조

```text
local-llm-wiki/
├── app/
│   ├── models/schema.py
│   ├── llm/base.py
│   ├── openai_client/client.py
│   ├── ollama/client.py
│   ├── pipeline/
│   │   ├── ingest.py
│   │   └── markdown.py
│   ├── query/query.py
│   ├── lint/check.py
│   ├── eval/benchmark.py
│   ├── eval/batch_benchmark.py
│   ├── utils/
│   └── outputs/benchmark/
├── config/ontology.yaml
├── config/models.txt          ← OpenAI MODEL:API_KEY_ENV
├── config/ollama-models.txt   ← Ollama model tags
├── config/loop.env            ← runtime/API key values
├── llmwiki/
│   ├── raw/
│   │   ├── articles/
│   │   ├── papers/
│   │   ├── repos/
│   │   ├── data/
│   │   ├── images/
│   │   └── assets/
│   ├── wiki/
│   │   ├── index.md
│   │   ├── log.md
│   │   ├── overview.md
│   │   ├── concepts/
│   │   ├── entities/
│   │   ├── sources/
│   │   ├── comparisons/
│   │   └── benchmarks/        ← Obsidian benchmark history
│   └── SCHEMA.md
└── tests/
```

## 책임 경계

```text
                 app/
                  │
          실행/변환/검증만 수행
                  │
                  ▼
          ┌─────────────────┐
          │    llmwiki/     │
          │                 │
          │ raw/            │ ← immutable source
          │ wiki/*.md       │ ← canonical knowledge
          └────────┬────────┘
                   │
            Open folder as Vault
                   ▼
               Obsidian
```

`app/obsidian`는 존재하지 않는다. Python 애플리케이션은 Obsidian API를 사용하지 않고 Markdown/YAML/wikilink만 생성한다.

## UUID 정책

LLM은 UUID를 생성하지 않는다.

```text
raw file bytes
      ↓
Source UUID (UUIDv5)
      ↓
Chunk UUID (Source UUID + stable chunk key)
      ↓
Knowledge UUID (type + normalized title)
      ↓
Relation UUID (source + relation + target + evidence)
```

따라서 Qwen3과 Gemma3를 번갈아 실행해도 같은 logical knowledge object의 identity가 흔들리지 않는다.

## 1단계 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Ollama

현재 benchmark 기본값:

```bash
ollama pull qwen3:4b-q4_K_M
ollama pull gemma3:4b-it-q4_K_M
```

학습/파인튜닝은 v4 범위에 포함하지 않는다. v4에서는 동일 원본을 여러 로컬 모델로 실행하고 품질/성능 차이를 측정한다.

## Ingest

원본을 `llmwiki/raw/papers/` 등에 넣는다.

```bash
python -m app.pipeline.ingest \
  --input ./llmwiki/raw/papers/transformer-paper.pdf \
  --llmwiki ./llmwiki \
  --model qwen3:4b-q4_K_M
```

실행 흐름:

```text
raw
 ↓
Docling DocumentConverter
 ↓
HierarchicalChunker
 ↓
Source/Chunk UUID
 ↓
기존 wiki/index.md + 관련 Markdown 참조
 ↓
Ollama structured output
 ↓
Pydantic validation
 ↓
ontology validation
 ↓
Markdown merge
 ↓
Obsidian [[wikilink]] 생성
 ↓
index.md / log.md 갱신
```

Docling의 `DocumentConverter`가 입력 문서를 `DoclingDocument`로 변환하고, Hierarchical/Hybrid chunking은 문서 구조를 기준으로 chunk를 만듭니다. v4 ingest는 먼저 안정적인 계층적 chunking으로 시작하며, 이후 필요하면 HybridChunker로 전환할 수 있다.

## Query

기본 query는 Markdown-first다.

```bash
python -m app.query.query "self attention"
```

QMD가 설치되어 있으면:

```bash
python -m app.query.query "self attention" --engine qmd
```

QMD는 별도 로컬 Markdown 검색 엔진으로 BM25/벡터/하이브리드 검색과 reranking을 제공하며, 프로젝트의 canonical Markdown을 대체하지 않는다. 현재 QMD 프로젝트는 `@tobilu/qmd`로 배포되며, v4는 이를 Python 의존성으로 묶지 않고 외부 CLI로 선택적으로 사용한다.

예:

```bash
npm install -g @tobilu/qmd
qmd collection add ./llmwiki/wiki --name llmwiki
qmd query "self attention" -n 10
```

## Lint

```bash
python -m app.lint.check
```

검사:

- provenance 누락
- relation source 불일치
- relation target 누락
- relation UUID 중복
- 허용되지 않은 ontology relation

## Obsidian

Obsidian에서 **`llmwiki/` 자체를 Vault**로 연다.

예를 들어 generated note는:

```markdown
## Relations

- `depends_on` → [[concepts/scaled-dot-product-attention|Scaled Dot Product Attention]]
```

형태를 사용한다.

따라서 Obsidian Graph View와 Backlinks는 별도 Python 플러그인 없이 Markdown wikilink를 통해 동작한다.

## Benchmark

Provider별 모델 목록은 코드에 하드코딩하지 않는다.

`config/models.txt`는 OpenAI 모델을 `MODEL:API_KEY_ENV` 형식으로 관리한다. 실제 API key 값은 이 파일에 쓰지 않고 `config/loop.env`의 해당 환경변수 이름에서 읽는다.

```text
gpt-5.6-luna:OPENAI_API_KEY
gpt-5.6-terra:OPENAI_API_KEY_TEAM_A
```

`config/ollama-models.txt`는 로컬 Ollama model tag를 한 줄에 하나씩 관리한다.

```text
qwen3:4b-instruct-2507-q4_K_M
gemma3:4b-it-q4_K_M
qwen3.5:4b
```

`config/loop.env` 예:

```env
OPENAI_API_KEY=
OPENAI_API_KEY_TEAM_A=
OPENAI_BASE_URL=
OLLAMA_HOST=http://localhost:3003/api/proxy
OLLAMA_API_KEY=
AUTO_RESUME_FAILED_RUN=true
```

`OPENAI_BASE_URL`은 공식 OpenAI API를 직접 사용할 때 비워둔다. 프록시/OpenAI-compatible gateway를 사용할 때만 지정한다.

Batch 실행 순서는 **OpenAI (`models.txt`) → Ollama (`ollama-models.txt`)**이며 모든 provider가 동일한 `chunks.jsonl`을 재사용한다.

순차 benchmark 실행:

```bash
python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/transformer-paper.pdf
```

동일한 원본에 대해 OpenAI 모델을 먼저 실행하고 Ollama 모델을 이어서 실행한다. OpenAI는 원격 inference이므로 VRAM cleanup을 하지 않으며, Ollama 모델이 끝난 뒤에만 `ollama stop <model>`과 VRAM 확인을 수행한다.

원시 실행 기록은 다음에 저장한다.

```text
app/outputs/benchmark/batch-runs/<run-id>/
├── 01-<model>.json
├── 01-<model>.md
├── 02-<model>.json
├── 02-<model>.md
└── ...
```

Obsidian에서도 benchmark 결과를 별도 영역으로 관리한다.

```text
llmwiki/wiki/benchmarks/
├── index.md
├── <run-id>.md
└── <run-id>/
    ├── 01-<model>.md
    ├── 02-<model>.md
    └── ...
```

따라서 `llmwiki/`를 Obsidian Vault로 열면 지식 문서와 benchmark 이력을 같은 Vault 안에서 분리해 볼 수 있다.

단일 모델 benchmark도 직접 실행할 수 있다.

```bash
python -m app.eval.benchmark \
  --input ./llmwiki/raw/papers/transformer-paper.pdf \
  --llmwiki ./llmwiki \
  --models qwen3:4b-q4_K_M gemma3:4b-it-q4_K_M
```

측정값:

- TTFT
- tokens/sec
- total duration
- Pydantic schema pass rate
- JSON recovery rate
- source citation preservation
- ontology compliance


## KDB / Graph DB는 왜 빠졌는가?

v4에서는 별도의 KDB를 만들지 않는다.

```text
Markdown = source of truth
Obsidian = human UI
QMD = optional search/index engine
Graph DB = future extension
```

이렇게 해야 `llmwiki/wiki/`만 백업해도 지식 전체를 복구할 수 있고, Docker에서 다음처럼 독립 Vault로 바로 마운트할 수 있다.

```bash
docker run --rm \
  -v ./llmwiki:/data/llmwiki \
  your-image
```

## 다음 레벨: 모델 학습/파인튜닝

v4에서는 하지 않는다.

먼저 다음을 안정화한다.

```text
raw → Docling → UUID → Ollama → ontology → wiki → Obsidian/QMD
```

그 후 v5에서 충분히 누적된 Wiki와 provenance를 학습 데이터셋으로 만들어:

```text
wiki + source evidence
        ↓
dataset construction
        ↓
SFT / LoRA / preference data
        ↓
local model training
```

으로 넘어간다.

## Sequential model batch benchmark

Run the same source document against multiple Ollama models sequentially. Each model gets its own Markdown report and JSON result. By default the runner stops the current model before starting the next one to release VRAM.

```bash
python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/attention-is-all-you-need.pdf \
  --llmwiki ./llmwiki \
  --models \
    qwen3:4b-instruct-2507-q4_K_M \
    qwen3:4b-q4_K_M \
    gemma3:4b-it-q4_K_M
```

For the 8 GB VRAM system, adding a larger model is also possible:

```bash
python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/attention-is-all-you-need.pdf \
  --models \
    qwen3.5:4b \
    qwen3.5:9b-q4_K_M \
    qwen3:4b-instruct-2507-q4_K_M \
    gemma3:4b-it-q4_K_M
```

Outputs are stored under `app/outputs/benchmark/batch-runs/<UTC-run-id>/` and a combined report is written to `app/outputs/benchmark/batch-<source>.md`.

## Benchmark Wiki outputs

`app.eval.batch_benchmark` now runs Docling/OCR once per batch, stores a shared `chunks.jsonl`, and reuses the same chunks for every model. Each model also produces an isolated Wiki sandbox under:

```text
llmwiki/benchmark-runs/<run_id>/<model-slug>/wiki/
```

The canonical `llmwiki/wiki/` is not modified by benchmark model outputs. Benchmark reports under `llmwiki/wiki/benchmarks/` include both runtime metrics and structural Wiki metrics. See `docs/BENCHMARK_WIKI_ARCHITECTURE.md`.
=======
# project01
프로젝트 실습과 과제 내용
>>>>>>> 60883cbe88d2d09e60bc85d68a90ad030bf663cb
