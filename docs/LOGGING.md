# WikiLLM 실행 로그

로깅은 `app/utils/logging.py`에서 중앙 설정합니다.

## 기본

`config/loop.env`의:

```env
LLMWIKI_LOG_LEVEL=INFO
```

를 사용합니다.

문제 추적을 강화하려면:

```env
LLMWIKI_LOG_LEVEL=DEBUG
```

로 바꿉니다.

## Batch benchmark

배치 실행 시:

```text
app/outputs/benchmark/batch-runs/<RUN_ID>/batch.log
```

에 배치 실행 로그가 기록됩니다.

각 모델은 별도 로그를 생성합니다.

```text
01-<model>.log
02-<model>.log
...
```

동시에 자식 `benchmark.py`의 stdout/stderr가 실행 중 터미널에 실시간으로 표시됩니다.

## 주요 단계

로그의 핵심 marker:

```text
MODEL BATCH START
MODEL START
CHUNK START
Ollama call start
Ollama call stream complete
Ollama call success
CHUNK LLM SUCCESS
CHUNK FAILED
MODEL END
MODEL BATCH RESULT
ollama stop
unload check
VRAM check
```

## Error code

현재 코드에서 주요 오류는 다음과 같이 표시합니다.

| Code | 의미 |
|---|---|
| `E_CONFIG_API_KEY_MISSING` | Ollama Admin API Key가 환경변수에 없음 |
| `E_OLLAMA_DEPENDENCY` | Python Ollama 패키지가 없음 |
| `E_OLLAMA_TRANSPORT` | Ollama/Proxy 요청 중 런타임 또는 네트워크 오류 |
| `E_RESPONSE_JSON` | 모델 응답을 JSON으로 해석할 수 없음 |
| `E_RESPONSE_SCHEMA` | JSON은 있으나 Pydantic schema 검증 실패 |
| `E_DOCLING_DEPENDENCY` | Docling 의존성 없음 |
| `E_DOCLING_CONVERSION` | Docling 문서 변환 실패 |
| `E_ONTOLOGY_VIOLATION` | ontology 규칙에 맞지 않는 relation |
| `E_WIKI_WRITE` | Wiki materialization 또는 파일 저장 실패 |
| `E_OUTPUT_MISSING` | benchmark 결과 파일이 생성되지 않음 |
| `E_OUTPUT_EMPTY` | benchmark 결과 파일이 비어 있음 |
| `E_BENCHMARK_SUBPROCESS` | 모델 benchmark 자식 프로세스가 비정상 종료 |
| `E_GPU_CLEANUP_TIMEOUT` | 모델 종료 후 GPU 메모리가 기준치 이하로 내려가지 않음 |

API Key 값 자체는 로그에 기록하지 않고 `SET` / `NOT_SET`만 표시합니다.
