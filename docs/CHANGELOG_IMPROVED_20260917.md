# Improved build — 2026-09-17

## Reliability / observability

- Added request IDs to Ollama calls.
- Added `OLLAMA HEARTBEAT` watchdog logs (`WAITING_FIRST_TOKEN` / `STREAMING`).
- Added batch `CHILD HEARTBEAT` when the child process is alive but silent.
- Added chunk/model `PROGRESS` logs with percent, chunks/min, elapsed time, and ETA.
- Added detailed load/prompt/decode metrics to logs and benchmark reports.
- Added top-5 slow chunk reporting.
- Added JSONL per-model checkpoints with resume support.

## Performance

- `FORCE_KEEP_ALIVE_ZERO` now defaults to `false` so the model remains loaded across chunks.
- Added optional bounded benchmark concurrency (`BENCHMARK_CONCURRENCY`, default `1`).
- Added optional short-chunk filtering (`MIN_LLM_CHARS`, default `0`).
- Cached Wiki title lookup during ingest rather than rescanning for every write.
- `run.sh` now uses `uv sync` instead of deleting and recreating `.venv` every run.

## Correctness

- Ontology compliance counts both invalid relations and unresolved relation endpoints.
- Fixed ingest chunk ordinal parsing from `source:0001` chunk keys.
- Corrected Pydantic validator type annotation for LLM-facing source refs.

## Security

- Removed hard-coded `OLLAMA_API_KEY` values from `config/loop.env` and `test.sh`.
- Runtime credentials now come from environment variables.
- Added `config/runtime.env.example` and ignored local runtime secret files.
