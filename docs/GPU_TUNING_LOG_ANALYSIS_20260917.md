# Qwen vs Gemma GPU/LLM log analysis — 2026-09-17

The uploaded Gemma log is partial, so the fair comparison uses chunks 1–187,
which are present in both logs. Qwen's full 1701-chunk completion is shown as a
separate reference.

| Metric | Qwen 1–187 | Gemma 1–187 | Gemma vs Qwen |
|---|---:|---:|---:|
| Mean TTFT | 341.50 ms | 481.09 ms | +40.87% |
| Median TTFT | 315.46 ms | 445.98 ms | +41.38% |
| P95 TTFT | 472.48 ms | 669.83 ms | +41.77% |
| Mean wall time/chunk | 2758.25 ms | 3649.61 ms | +32.32% |
| Median wall time/chunk | 1610.16 ms | 2427.20 ms | +50.74% |
| Mean decode speed | 84.84 tok/s | 75.63 tok/s | -10.86% |
| Mean prompt eval time | 157.98 ms | 259.34 ms | +64.16% |
| Mean output tokens | 208.52 | 244.49 | +17.25% |
| Median output tokens | 109 | 146 | +33.94% |
| P95 output tokens | 755.9 | 577.2 | -23.64% |
| Max output tokens | 2196 | 4192 | +90.89% |
| Total output tokens | 38,994 | 45,720 | +17.25% |
| Ontology errors | 3 | 386 | +383 |
| Chunks with ontology errors | 2 / 187 | 110 / 187 | much higher |
| Citation preserved | 100% | 100% | same |
| Effective throughput (sum wall) | 21.75 chunks/min | 16.44 chunks/min | -24.42% |

Qwen completed the full 1701 chunks at about 30.07 chunks/min in the runtime
progress log. Gemma was around 16 chunks/min near chunk 170 in the partial log.

## Interpretation

Gemma is already 100% GPU according to `ollama ps`, so CPU offload is not the
main problem. The dominant issue in this run is model behavior: Gemma sometimes
generates very large structured responses. For the same chunk 7, Gemma emitted
4192 output tokens and took about 54.4 seconds, while Qwen emitted 685 tokens and
took about 8.5 seconds. Large Gemma outputs also frequently contain unresolved
relation endpoints.

Increasing `num_ctx` would consume more VRAM but would not solve this decode and
over-generation bottleneck. Therefore this patch keeps `num_ctx=8192`, adds
per-model concurrency support, adds a GPU-tuned Gemma profile with concurrency=2,
and adds relation endpoint normalization plus output-token warnings.

## Runtime profiles

Default/resume-safe profile:

```text
config/ollama-models.txt
```

GPU experiment profile:

```text
config/ollama-models.gpu-tuned.txt
```

Run a new clean GPU-tuned benchmark with:

```bash
uv run python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/llmops-managing-llm-in-production.pdf \
  --llmwiki ./llmwiki \
  --ollama-model-list ./config/ollama-models.gpu-tuned.txt \
  --no-auto-resume-failed
```

Do not mix an existing concurrency=1 checkpoint with concurrency=2 metrics.
The benchmark now records runtime parameters in checkpoint metadata and archives
incompatible checkpoints before starting a clean measurement.

`num_predict` is supported as an optional per-model setting, but it is not set
in the recommended profile because a hard output-token cap can truncate JSON and
turn a long response into a schema failure. First use the relation filter and
concurrency experiment; only add `num_predict` after measuring an appropriate
safe ceiling.
