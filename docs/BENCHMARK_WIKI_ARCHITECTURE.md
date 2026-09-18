# Benchmark + Wiki Architecture

## Goal

Benchmark mode evaluates not only runtime metrics but the actual Wiki knowledge base produced by each model.
The canonical `llmwiki/wiki/` is never modified by benchmark model outputs.

## Flow

```text
raw source
  -> Docling/OCR once
  -> shared chunks.jsonl
  -> Model A -> isolated Wiki A
  -> Model B -> isolated Wiki B
  -> ...
  -> benchmark metrics + Wiki structural quality metrics
```

## Output layout

```text
llmwiki/
  wiki/                         # canonical human-reviewed knowledge
    benchmarks/                 # benchmark reports only
  benchmark-runs/
    <run_id>/
      <model-slug>/
        README.md
        wiki/
          concepts/
          entities/
          sources/
          comparisons/
          index.md
          log.md
```

Raw execution artifacts and the shared chunk cache remain under:

```text
app/outputs/benchmark/batch-runs/<run_id>/
```

## Automatic Wiki metrics

The benchmark records descriptive structural metrics: document and relation counts, relation-target integrity, source/provenance coverage, chunk-reference coverage, orphan-document rate, duplicate titles, and relation density. These do not by themselves prove factual correctness. Human review or a separate grounded evaluator is still required for faithfulness, usefulness, concept granularity, and merge quality.

## Promotion rule

Benchmark Wikis are candidates, not canonical knowledge. Review a model Wiki first, then explicitly merge/promote selected knowledge into `llmwiki/wiki/`.
