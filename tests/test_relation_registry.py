from uuid import uuid4

from app.models.schema import (
    ConceptDraft,
    DocumentType,
    KnowledgeDocument,
    KnowledgeExtraction,
    RelationDraft,
    RelationType,
)
from app.pipeline.ingest import _name_key, knowledge_uuid, load_ontology, materialize_relations


def test_relation_can_target_existing_wiki_document() -> None:
    target = KnowledgeDocument(
        id=knowledge_uuid(DocumentType.CONCEPT, "Monitoring"),
        type=DocumentType.CONCEPT,
        title="Monitoring",
        summary="existing",
    )
    extraction = KnowledgeExtraction(
        title="chunk",
        concepts=[ConceptDraft(name="LLMOps", summary="ops")],
        relations=[RelationDraft(source="LLMOps", relation=RelationType.USES, target="Monitoring")],
    )
    known_docs = {target.id: target}
    known_names = {_name_key(target.title): (target.id, target.type)}
    ontology = load_ontology(__import__('pathlib').Path('config/ontology.yaml'))
    docs, errors = materialize_relations(
        extraction,
        ontology,
        known_documents=known_docs,
        known_names=known_names,
    )
    assert errors == []
    llmops = next(doc for doc in docs if doc.title == "LLMOps")
    assert llmops.relations[0].target == target.id
