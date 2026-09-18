# WikiLLM benchmark quality evaluator integration

Add these files to the project:

```text
app/eval/quality_eval.py
app/eval/evaluate_run.py
app/eval/golden/sample.example.json
```

## Run manually

```bash
uv run python -m app.eval.evaluate_run \
  --run-dir app/outputs/benchmark/batch-runs/<RUN_ID> \
  --golden-dir app/eval/golden
```

Outputs:

```text
<run-dir>/quality-summary.json
<run-dir>/quality-summary.md
```

## Optional automatic execution from batch_benchmark.py

Add this import near the top:

```python
from app.eval.evaluate_run import evaluate_batch_run
```

After `write_overview_note(...)` and before the final success-count/return section, add:

```python
    try:
        quality_rows = evaluate_batch_run(
            run_dir=run_dir,
            golden_dir=PROJECT_ROOT / "app" / "eval" / "golden",
        )
        logger.info("Quality evaluation complete models=%d", len(quality_rows))

        quality_md = run_dir / "quality-summary.md"
        if quality_md.exists():
            obsidian_quality = obsidian_dir / run_id / "quality-summary.md"
            obsidian_quality.parent.mkdir(parents=True, exist_ok=True)
            obsidian_quality.write_text(
                quality_md.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    except Exception:
        # Quality evaluation must not invalidate a completed inference benchmark.
        logger.exception("QUALITY EVALUATION FAILED")
```

## Golden sample

Replace `REPLACE-WITH-REAL-CHUNK-UUID` with an actual chunk UUID from `chunks.jsonl` / checkpoint output.
Create 20-50 representative golden samples first; 50-100 can be used for a stronger benchmark.

The evaluator computes:

- JSON parse rate
- schema pass rate
- citation presence/preservation rate
- ontology compliance rate
- relation endpoint validity rate
- concept precision / recall / F1 (when golden data exists)
- relation precision / recall / F1 (when golden data exists)
- structural score
- knowledge score

Performance metrics already produced by `benchmark.py` are kept separate (TTFT, tokens/sec, total duration, JSON recovery).
