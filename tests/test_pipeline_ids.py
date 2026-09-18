from pathlib import Path

from app.pipeline.ingest import chunk_uuid, knowledge_uuid, load_ontology, source_id_from_bytes
from app.models.schema import DocumentType, RelationType


def test_source_uuid_is_deterministic() -> None:
    payload = b"immutable source"
    assert source_id_from_bytes(payload) == source_id_from_bytes(payload)


def test_knowledge_uuid_is_deterministic() -> None:
    a = knowledge_uuid(DocumentType.CONCEPT, "Self-Attention")
    b = knowledge_uuid(DocumentType.CONCEPT, " self-attention ")
    assert a == b


def test_chunk_and_relation_uuid_are_deterministic() -> None:
    source = source_id_from_bytes(b"source")
    chunk_a = chunk_uuid(source, "doc:0001")
    chunk_b = chunk_uuid(source, "doc:0001")
    assert chunk_a == chunk_b


def test_ontology_has_type_constraints() -> None:
    ontology = load_ontology(Path("config/ontology.yaml"))
    assert ontology.relations["depends_on"].source == ["concept", "entity"]
    assert ontology.relations["depends_on"].target == ["concept", "entity"]
