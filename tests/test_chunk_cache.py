from pathlib import Path
from uuid import uuid4

from app.pipeline.chunk_cache import load_chunk_cache, write_chunk_cache
from app.pipeline.ingest import DocumentChunk


def test_chunk_cache_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    sid = uuid4()
    chunk = DocumentChunk(id=uuid4(), chunk_id="source:0000", source_id=sid, text="hello", meta={"x": 1})
    # Cache validation uses the content-derived source id, so use the real source id in the record.
    from app.pipeline.ingest import source_id
    chunk.source_id = source_id(source)
    path = tmp_path / "chunks.jsonl"
    write_chunk_cache(path, source, [chunk])
    loaded = load_chunk_cache(path, source)
    assert len(loaded) == 1
    assert loaded[0].text == "hello"
    assert loaded[0].id == chunk.id
