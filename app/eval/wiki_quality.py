from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import UUID

from app.models.schema import DocumentType, WikiQualityMetrics
from app.pipeline.markdown import load_knowledge_documents


def evaluate_wiki(wiki_root: Path, selected_chunk_ids: set[UUID] | None = None) -> WikiQualityMetrics:
    docs = [doc for doc, _ in load_knowledge_documents(wiki_root)]
    by_id = {doc.id: doc for doc in docs}
    types = Counter(doc.type for doc in docs)
    title_counts = Counter(doc.title.strip().casefold() for doc in docs if doc.title.strip())

    relations = [rel for doc in docs for rel in doc.relations]
    unresolved = sum(1 for rel in relations if rel.target not in by_id)
    relation_integrity = 1.0 if not relations else max(0.0, (len(relations) - unresolved) / len(relations))

    knowledge_docs = [doc for doc in docs if doc.type != DocumentType.SOURCE]
    sourced = sum(1 for doc in knowledge_docs if doc.sources)
    sourced_rate = 1.0 if not knowledge_docs else sourced / len(knowledge_docs)

    referenced_chunks: set[UUID] = set()
    for doc in docs:
        referenced_chunks.update(ref.chunk_id for ref in doc.sources)
        for rel in doc.relations:
            referenced_chunks.update(ref.chunk_id for ref in rel.source_refs)
    if selected_chunk_ids:
        covered = len(referenced_chunks & selected_chunk_ids)
        chunk_coverage = covered / len(selected_chunk_ids)
    else:
        chunk_coverage = 0.0

    connected: set[UUID] = set()
    for rel in relations:
        connected.add(rel.source)
        connected.add(rel.target)
    orphan = sum(1 for doc in knowledge_docs if doc.id not in connected)
    orphan_rate = 0.0 if not knowledge_docs else orphan / len(knowledge_docs)

    return WikiQualityMetrics(
        documents=len(docs),
        concepts=types[DocumentType.CONCEPT],
        entities=types[DocumentType.ENTITY],
        sources=types[DocumentType.SOURCE],
        comparisons=types[DocumentType.COMPARISON],
        relations=len(relations),
        unresolved_relation_targets=unresolved,
        relation_integrity_rate=relation_integrity,
        sourced_document_rate=sourced_rate,
        referenced_chunks=len(referenced_chunks),
        chunk_reference_coverage=chunk_coverage,
        orphan_documents=orphan,
        orphan_document_rate=orphan_rate,
        duplicate_titles=sum(count - 1 for count in title_counts.values() if count > 1),
        average_relations_per_document=(len(relations) / len(knowledge_docs)) if knowledge_docs else 0.0,
    )
