from uuid import uuid4

from app.eval.benchmark import _normalize_model_relations
from app.models.schema import ConceptDraft, KnowledgeExtraction, RelationDraft
from app.pipeline.ingest import load_ontology
from app.utils.paths import PROJECT_ROOT


def test_normalize_relations_keeps_cross_chunk_nodes_and_drops_unknown_endpoint() -> None:
    a = uuid4()
    b = uuid4()
    extractions = {
        a: KnowledgeExtraction(
            title="chunk-a",
            concepts=[ConceptDraft(name="Alpha", summary="alpha")],
            entities=[],
            relations=[
                RelationDraft(source="Alpha", relation="related_to", target="Beta"),
                RelationDraft(source="Alpha", relation="related_to", target="Never Extracted"),
            ],
        ),
        b: KnowledgeExtraction(
            title="chunk-b",
            concepts=[ConceptDraft(name="Beta", summary="beta")],
            entities=[],
            relations=[],
        ),
    }
    ontology = load_ontology(PROJECT_ROOT / "config" / "ontology.yaml")
    cleaned, errors, raw_count, retained_count = _normalize_model_relations(extractions, ontology)

    assert raw_count == 2
    assert retained_count == 1
    assert len(cleaned[a].relations) == 1
    assert cleaned[a].relations[0].target == "Beta"
    assert errors[a] == []
