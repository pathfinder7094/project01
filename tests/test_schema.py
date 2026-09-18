
from app.models.schema import (
    DocumentType,
    KnowledgeExtraction,
    KnowledgeDocument,
    Relation,
    RelationType,
    SourceRef,
)


def test_llm_schema_is_id_free_but_validates_provenance_ids() -> None:
    payload = {
        "title": "Attention",
        "language": "en",
        "abstract": "Contextual token interaction.",
        "concepts": [
            {
                "name": "Attention",
                "summary": "A weighted interaction mechanism.",
                "definition": "",
                "aliases": [],
                "source_refs": [
                    {
                        "source_file": "raw/example.pdf",
                        "chunk_key": "example:0001",
                    }
                ],
                "confidence": 0.9,
            }
        ],
        "entities": [],
        "relations": [],
        "source_refs": [],
        "wiki_paths": ["concepts/attention.md"],
    }
    parsed = KnowledgeExtraction.model_validate(payload)
    assert parsed.concepts[0].source_refs[0].chunk_key == "example:0001"


def test_materialized_uuid_identity() -> None:
    document = KnowledgeDocument(
        type=DocumentType.CONCEPT,
        title="Self-Attention",
        summary="A concept.",
    )
    from uuid import UUID
    assert isinstance(document.id, UUID)


def test_relation_uuid_model() -> None:
    from uuid import UUID
    relation = Relation(
        source=UUID("550e8400-e29b-41d4-a716-446655440000"),
        target=UUID("8f14e45f-ea2d-4d7c-9f7a-123456789abc"),
        relation=RelationType.USES,
    )
    assert isinstance(relation.id, UUID)
