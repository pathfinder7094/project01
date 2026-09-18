from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from app.pipeline.ingest import DocumentChunk, chunk_document, source_id
from app.utils.logging import configure_logging, get_logger

CACHE_VERSION = 1


def write_chunk_cache(path: Path, source: Path, chunks: list[DocumentChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_type": "meta",
        "version": CACHE_VERSION,
        "source_id": str(source_id(source)),
        "source_file": str(source.resolve()),
        "chunks": len(chunks),
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for chunk in chunks:
            handle.write(json.dumps({
                "_type": "chunk",
                "id": str(chunk.id),
                "chunk_id": chunk.chunk_id,
                "source_id": str(chunk.source_id),
                "text": chunk.text,
                "meta": chunk.meta,
            }, ensure_ascii=False) + "\n")


def load_chunk_cache(path: Path, source: Path | None = None) -> list[DocumentChunk]:
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[DocumentChunk] = []
    meta: dict[str, Any] | None = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        if item.get("_type") == "meta":
            meta = item
            continue
        if item.get("_type") != "chunk":
            continue
        chunks.append(DocumentChunk(
            id=UUID(item["id"]),
            chunk_id=str(item["chunk_id"]),
            source_id=UUID(item["source_id"]),
            text=str(item["text"]),
            meta=dict(item.get("meta") or {}),
        ))
    if meta is None:
        raise ValueError(f"Chunk cache metadata missing: {path}")
    if int(meta.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported chunk cache version: {meta.get('version')}")
    if source is not None and str(source_id(source)) != str(meta.get("source_id")):
        raise ValueError(f"Chunk cache source mismatch: {path}")
    return chunks


def ensure_chunk_cache(path: Path, source: Path, force: bool = False) -> list[DocumentChunk]:
    logger = get_logger(__name__)
    if path.exists() and not force:
        try:
            chunks = load_chunk_cache(path, source)
            logger.info("CHUNK CACHE HIT path=%s chunks=%d", path, len(chunks))
            return chunks
        except Exception as exc:
            logger.warning("CHUNK CACHE INVALID path=%s error=%s; rebuilding", path, exc)
    logger.info("CHUNK CACHE BUILD source=%s path=%s", source, path)
    chunks = chunk_document(source)
    write_chunk_cache(path, source, chunks)
    logger.info("CHUNK CACHE READY path=%s chunks=%d", path, len(chunks))
    return chunks


def main() -> None:
    configure_logging(component=__name__)
    parser = argparse.ArgumentParser(description="Create/reuse a Docling chunk cache")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    chunks = ensure_chunk_cache(output, source, force=args.force)
    print(json.dumps({"source": str(source), "cache": str(output), "chunks": len(chunks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
