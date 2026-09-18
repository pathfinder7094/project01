# LLM Wiki Schema v4

## 1. Canonical layers

### raw/

원본 PDF, 웹 문서, Markdown, 저장소, 이미지 등을 저장한다.

- pipeline은 읽기 전용으로 취급한다.
- 원본 파일을 수정하지 않는다.
- 모든 생성 지식은 가능한 한 raw source로 역추적 가능해야 한다.

### wiki/

모델과 사람이 함께 축적하는 **canonical knowledge layer**다.

```text
wiki/
├── index.md
├── log.md
├── overview.md
├── concepts/
├── entities/
├── sources/
└── comparisons/
```

`wiki/*.md`가 지식의 source of truth다.

## 2. UUID identity

Pipeline이 모든 영속 객체의 UUID를 관리한다.

- Source: source bytes hash 기반 UUIDv5
- Chunk: Source UUID + stable chunk key 기반 UUIDv5
- Knowledge: document type + normalized title 기반 UUIDv5
- Relation: source UUID + relation type + target UUID + evidence 기반 UUIDv5

LLM은 UUID를 생성하지 않는다.

파일명은 사람이 읽기 위한 이름이고 UUID는 시스템 식별자다.

## 3. Markdown frontmatter

Concept/Entity/Comparison:

```yaml
---
id: <knowledge-uuid>
type: concept
title: Self-Attention
aliases: []
summary: "..."
sources:
  - source_id: <source-uuid>
    chunk_id: <chunk-uuid>
    source_file: raw/papers/attention.pdf
    locator: "page 3 / Attention"
    quote: "..."
relations:
  - id: <relation-uuid>
    source: <knowledge-uuid>
    type: depends_on
    target: <target-knowledge-uuid>
    evidence: "..."
    source_refs: []
---
```

## 4. Ontology

허용 관계와 source/target type constraint는 `config/ontology.yaml`이 정의한다.

예:

```text
Concept --depends_on--> Concept
Entity  --instance_of--> Concept
```

임의 관계명을 생성하지 않는다.

## 5. Benchmark notes in Obsidian

`wiki/benchmarks/` contains operational benchmark history. These notes are not canonical knowledge objects and are not merged into the concept/entity graph. They are generated separately by the batch benchmark runner so model performance can be reviewed directly in Obsidian.

## 6. Obsidian

Python 실행 엔진은 Obsidian API에 의존하지 않는다.

관계는 frontmatter의 typed UUID edge와 본문의 Markdown wikilink로 동시에 표현한다.

```markdown
## Relations

- `depends_on` → [[concepts/scaled-dot-product-attention|Scaled Dot Product Attention]]
```

이 구조에서 Obsidian Graph View와 Backlinks는 Markdown wikilink를 이용한다.

## 7. Query

v4의 기본 query는 Markdown-first다.

```text
query → wiki/*.md
```

QMD가 설치되어 있으면 외부 검색 엔진으로:

```text
query → QMD → wiki/*.md
```

를 사용한다.

QMD의 검색 인덱스는 canonical knowledge를 대체하지 않는다.

## 8. Merge

새 원본을 ingest할 때:

1. source/chunk UUID를 생성한다.
2. `wiki/index.md`와 관련 기존 note를 탐색한다.
3. 동일 knowledge UUID는 merge한다.
4. 신규 provenance는 누적한다.
5. 동일 relation UUID는 다시 쓰지 않는다.
6. 기존 지식을 무단 삭제하지 않는다.

## 9. Language policy

- 한국어 기술 설명은 한국어를 유지한다.
- 국제적으로 고정된 기술 용어는 English를 유지한다.
- 원문의 핵심 terminology와 고유명사는 불필요하게 번역하지 않는다.
- 의미를 바꾸는 의역을 피한다.
- 파일명은 kebab-case를 사용하며 한글을 허용한다.

## 10. Query / Lint / Ingest

### Ingest

```text
raw → Docling → chunks → Ollama → Pydantic → ontology → wiki merge
```

### Query

```text
wiki → local search or QMD → relevant notes
```

### Lint

- provenance 누락
- relation source 불일치
- target 누락
- relation UUID 중복
- ontology relation 위반

## 11. No KDB in v4

v4에서는 별도의 SQLite/Neo4j KDB를 canonical storage로 사용하지 않는다.

```text
Markdown = canonical knowledge
Obsidian = UI
QMD = optional search/index
Graph DB = future extension
```

필요해지면 향후 `derived/` 아래에 재생성 가능한 graph/vector index를 추가한다.

## 12. Git

권장 추적:

```text
llmwiki/raw/
llmwiki/wiki/
llmwiki/SCHEMA.md
config/ontology.yaml
```

검색/벡터/그래프 파생 데이터는 필요에 따라 별도 ignore한다.

## 13. Training boundary

v4는 inference/extraction/organization/benchmark 단계다.

SFT, LoRA, preference optimization 등의 모델 학습은 v5에서 다룬다.
