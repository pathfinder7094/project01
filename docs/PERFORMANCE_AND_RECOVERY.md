# Benchmark performance and recovery

## What changed

- `FORCE_KEEP_ALIVE_ZERO=false` by default: a model stays loaded across chunks and is stopped after its model run.
- `OLLAMA HEARTBEAT`: logs whether a request is waiting for the first token or streaming.
- `CHILD HEARTBEAT`: the batch parent reports when the benchmark subprocess is alive but quiet.
- Detailed timings: model load, prompt evaluation, output evaluation, token counts, TTFT, wall time, and tokens/sec.
- `PROGRESS`: completed chunks, success/failure counts, chunks/minute, elapsed time, and ETA.
- Per-model JSONL checkpoints. Re-running the same benchmark/report path can resume completed chunks.
- Optional bounded benchmark concurrency. Default is `1`; test `2` only after checking VRAM and TTFT.
- Ontology compliance now counts unresolved endpoints as validation errors.
- Ingest chunk ordinal parsing fixed (`source:0001`).
- Wiki title lookup is cached during an ingest run instead of rescanning the whole wiki for every document write.

## Recommended first run

`config/loop.env`에 필요한 provider credential을 설정한다.

```env
OLLAMA_HOST=http://localhost:3003/api/proxy
OLLAMA_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

그 다음 실행한다.

```bash
uv sync
uv run python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/LLMOps-Managing-Large-Language-Models-in-Production.pdf \
  --llmwiki ./llmwiki
```

Keep `BENCHMARK_CONCURRENCY=1` for the first baseline. After confirming load time is near zero on subsequent chunks and VRAM is stable, try `BENCHMARK_CONCURRENCY=2`.

## Failed-run auto resume

`batch_benchmark.py` now reuses the latest incomplete run for the same source when `AUTO_RESUME_FAILED_RUN=true` (default).
It reuses `chunks.jsonl`, metric checkpoints, and structured extraction journals, so completed chunks are not sent to the model again.

Explicit recovery is also available:

```bash
uv run python -m app.eval.batch_benchmark \
  --input <same-source.pdf> \
  --llmwiki ./llmwiki \
  --resume-run-id 20260917T091553Z
```

If every selected chunk has both a successful metric and a structured extraction, the model client is never instantiated; the process proceeds directly to Wiki materialization/report generation.

Wiki filenames use UUID identity as the authoritative key. If different documents normalize to the same readable slug, a deterministic UUID suffix is added instead of aborting the whole Wiki build.


## OpenAI + Ollama provider order

- `config/models.txt`: OpenAI `MODEL:API_KEY_ENV` mappings.
- `config/ollama-models.txt`: Ollama model tags.
- OpenAI targets run first, Ollama targets second.
- Both reuse the same Docling `chunks.jsonl`.
- Checkpoint lookup is provider/model based and can reuse legacy pre-provider paths such as `01-qwen....checkpoints` even when OpenAI insertion changes the current sequence number.
