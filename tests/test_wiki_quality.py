from pathlib import Path
from uuid import uuid4

from app.eval.wiki_quality import evaluate_wiki
from app.models.schema import DocumentType, KnowledgeDocument, SourceRef
from app.pipeline.ingest import upsert_document
from app.utils.paths import ensure_layout


def test_wiki_quality_counts_documents_and_sources(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    source_id = uuid4()
    chunk_id = uuid4()
    doc = KnowledgeDocument(
        id=uuid4(),
        type=DocumentType.CONCEPT,
        title="Test Concept",
        summary="summary",
        sources=[SourceRef(source_id=source_id, chunk_id=chunk_id, source_file="raw/test.txt")],
    )
    upsert_document(tmp_path, doc)
    quality = evaluate_wiki(tmp_path / "wiki", {chunk_id})
    assert quality.documents == 1
    assert quality.concepts == 1
    assert quality.sourced_document_rate == 1.0
    assert quality.chunk_reference_coverage == 1.0
