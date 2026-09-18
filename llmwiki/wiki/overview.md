---
id: 00000000-0000-0000-0000-000000000001
type: overview
title: LLM Wiki Overview
---

# LLM Wiki Overview

이 문서는 위키를 처음 탐색할 때 읽는 진입점이다.

## Navigation

- [[index]] — 전체 지식 목록
- [[log]] — pipeline 작업 이력

## Layers

- `raw/` — 불변 원본
- `wiki/` — canonical knowledge
- Obsidian — `llmwiki/`를 여는 사용자 인터페이스
- QMD — 선택적 Markdown 검색/인덱스 엔진

## Workflow

`ingest → query → lint`

## UUID

Source → Chunk → Knowledge → Relation의 identity는 pipeline이 UUID로 관리한다.

## Training Boundary

v4는 모델 학습이 아니라 지식 추출, 누적, 검색, 검증 및 benchmark에 집중한다.
모델 학습/SFT/LoRA는 다음 단계에서 별도로 설계한다.
