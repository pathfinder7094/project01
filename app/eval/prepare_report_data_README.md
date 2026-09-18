# `prepare_report_data.py` 적용 방법

이 스크립트는 **기존 WikiLLM benchmark를 다시 실행하지 않고** 이미 생성된 `chunks.jsonl`, 모델별 `*.json`, `*.log`, `*.checkpoints/`, 생성 Wiki를 읽어서 제출/공유 가능한 작은 결과 패키지를 만듭니다.

## 1. 파일 배치

```text
<project-root>/
└── app/
    └── eval/
        └── prepare_report_data.py
```

## 2. 현재 run 기준 실행

프로젝트 루트에서:

```bash
uv run python -m app.eval.prepare_report_data \
  --run-dir app/outputs/benchmark/batch-runs/20260917T091553Z \
  --llmwiki /mnt/c/Users/pathf/llmwiki
```

Golden dataset이 있으면:

```bash
uv run python -m app.eval.prepare_report_data \
  --run-dir app/outputs/benchmark/batch-runs/20260917T091553Z \
  --llmwiki /mnt/c/Users/pathf/llmwiki \
  --golden-dir app/eval/golden
```

Cloud benchmark 결과가 별도 디렉터리에 있으면 coverage에 포함하려고 경로를 넘길 수 있습니다.

```bash
  --cloud-dir app/outputs/benchmark/cloud/<RUN_ID>
```

## 3. 생성 파일

기본 출력:

```text
app/outputs/report-data/20260917T091553Z/
├── report-data.json       # 가장 중요한 공유 파일
├── model-comparison.csv   # 표/차트 작성용
├── evidence.jsonl         # 대표 오류·느린 chunk·retry 사례만 선별
├── report-summary.md      # 사람이 바로 읽는 요약
└── share-package.zip      # 위 4개만 묶은 공유 패키지
```

원본 `chunks.jsonl`, 수천 개 checkpoint, 수 MB log, Wiki Markdown 전체를 공유하지 않아도 됩니다.

## 4. 계산되는 주요 지표

### Dataset
- chunk 수 / unique ID / 중복 ID
- 문자 길이 mean / median / std / min / max / p90 / p95 / p99
- 20/50/100자 미만 chunk 수
- chunk type 분포

### Performance
- TTFT / wall / total / load duration 분포
- tokens/sec 분포
- prompt/output token 총량
- weighted tokens/sec
- chunks/min, elapsed time
- first 20 chunks vs 나머지 warm-up 비교
- chunk 길이 ↔ TTFT/total time/output token Pearson correlation
- chunk 길이 사분위별 성능

### Reliability
- inference 성공률
- 실패 수
- retry rate
- schema pass rate
- JSON recovery rate
- inference 완료와 end-to-end pipeline 완료를 별도 구분

### Knowledge / Wiki quality
- citation preservation
- ontology compliance
- unresolved relation endpoint 수
- relation endpoint validity (분모를 확보할 수 있을 때만)
- concept mentions / unique normalized concepts
- relation type 분포
- Wiki 문서 수 / 폴더 분포
- wikilink / broken link / valid link rate
- orphan node / untitled / duplicate stem
- frontmatter id/type/title 누락
- ontology에 없는 relation type

### Golden (있을 때만)
- dev/test/all split
- concept micro precision / recall / F1
- relation micro precision / recall / F1
- macro F1

### Cross-model insight
- 동일 chunk의 concept Jaccard
- concept exact agreement
- relation Jaccard
- relation exact agreement

이 값은 정확도가 아니라 **모델 간 불일치가 큰 검토 대상 chunk를 찾는 지표**입니다.

### Environment / model metadata
- Python/platform
- 현재 `nvidia-smi` GPU/RAM snapshot
- Ollama version
- `ollama show <model>` 결과
- model tag에서 parameter/quantization 힌트 추출
- benchmark-time VRAM은 log에 `size_vram`/VRAM 값이 실제 기록된 경우에만 측정치로 사용

## 5. 프로젝트 요구사항 coverage 체크

`report-data.json`은 요구사항별로 `available / partial / missing`을 기록합니다.

특히 현재 WikiLLM run이 `chunks.jsonl` 기반이면 원 프로젝트 문서의 "고정 질문 10개 × 2회" 프로토콜을 충족했다고 자동 간주하지 않고 `not_demonstrated_by_chunk_run`으로 표시합니다.

Cloud 결과, 비용, authoritative ModelCard/license, 사람이 직접 매긴 품질점수 등 실제 데이터가 없으면 값을 만들어내지 않고 `missing` 또는 `partial`로 남깁니다.

## 6. 공유할 파일

가장 간단하게는 다음 하나만 공유하면 됩니다.

```text
share-package.zip
```

세부 원인 확인이 필요할 때만 원본 log/checkpoint를 추가로 제공하면 됩니다.
