# QMD integration (optional)

QMD is not part of the Python environment. It is an external local Markdown search engine.

Official project: https://github.com/tobi/qmd

The current QMD project combines keyword/BM25 and vector/hybrid search with local reranking. The v4 Python query module only invokes the `qmd` executable when it is installed and otherwise falls back to a simple local Markdown search.

Example:

```bash
npm install -g @tobilu/qmd
qmd collection add ./llmwiki/wiki --name llmwiki
qmd query "self attention" -n 10
```

Do not treat the QMD index as canonical data. The Markdown files remain the source of truth.
