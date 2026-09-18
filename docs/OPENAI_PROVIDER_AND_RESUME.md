# OpenAI provider + checkpoint resume

## Model configuration

`config/models.txt`:

```text
gpt-5.6-luna:OPENAI_API_KEY
```

`config/ollama-models.txt`:

```text
qwen3:4b-instruct-2507-q4_K_M
gemma3:4b-it-q4_K_M
```

`config/loop.env` contains the actual values. `models.txt` contains only the environment-variable name, never the secret itself.

## Execution

```text
PDF -> Docling/OCR once -> chunks.jsonl
                         -> OpenAI model(s) -> isolated provider/model Wiki
                         -> Ollama model(s) -> isolated provider/model Wiki
```

OpenAI uses the SDK default endpoint when `OPENAI_BASE_URL` is empty. Set it only for a proxy or OpenAI-compatible gateway.

## Resume

`AUTO_RESUME_FAILED_RUN=true` reuses the latest incomplete run for the same source content. `--resume-run-id <run-id>` selects one explicitly.

Checkpoint directories are matched by provider/model rather than only by sequence. This preserves compatibility with older runs where an Ollama model was sequence 1 before OpenAI targets were inserted before it.

If metrics and structured extraction journals are complete, no LLM client is created and the process goes directly to Wiki materialization/report generation.
