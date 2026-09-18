#!/bin/bash

PDF_FILE_NAME="pdf.pdf"
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# # uv run pytest



uv run python -m app.eval.batch_benchmark \
  --input ./llmwiki/raw/papers/$PDF_FILE_NAME \
  --llmwiki ./llmwiki