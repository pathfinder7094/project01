# Benchmark Wiki improvement - 2026-09-17

- Benchmark mode now produces an isolated full Wiki for each model under `llmwiki/benchmark-runs/<run_id>/<model>/wiki/`.
- Canonical `llmwiki/wiki/` is not modified by benchmark model output.
- Added descriptive Wiki quality metrics: document/relation counts, provenance coverage, chunk-reference coverage, relation integrity, orphan rate, duplicate titles, and relation density.
- Added shared Docling chunk cache. Batch preprocessing runs once in a short-lived subprocess and all model subprocesses reuse the same `chunks.jsonl`. This prevents repeated OCR/Docling work and releases preprocessing RAM before Ollama inference starts.
- Benchmark checkpoints now have a parallel structured-extraction journal so a model Wiki can be rebuilt on resume.
- Relation materialization can resolve endpoints against knowledge already present in the Wiki registry, reducing false unresolved-endpoint warnings across chunks/documents.
- `config/loop.env` values, including the configured Ollama proxy key, are forwarded to child benchmark processes.
- Added API-key preflight before Docling/OCR to avoid expensive preprocessing when runtime configuration is incomplete.
- Added tests for chunk cache, Wiki quality metrics, and cross-Wiki relation resolution.
