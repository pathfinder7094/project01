#!/bin/bash
#
#
#
#


uv run python -m app.eval.prepare_report_data \
  --run-dir app/outputs/benchmark/batch-runs/20260917T121601Z \
  --llmwiki ./llmwiki
