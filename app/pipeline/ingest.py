from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import yaml
from pydantic import BaseModel

from app.models.schema import (
    ChunkRecord,
    ConceptDraft,
    DocumentType,
    EntityDraft,
    KnowledgeDocument,
    KnowledgeExtraction,
    Relation,
    RelationDraft,
    RelationType,
    SourceRef,
    SourceRefDraft,
)
from app.ollama.client import OllamaClient
from app.utils.paths import APP_ROOT, DEFAULT_LLMWIKI_ROOT, ensure_layout, resolve_root
from app.utils.slugs import stable_slug
from app.pipeline.markdown import document_from_markdown, load_knowledge_documents, write_index
from app.utils.logging import configure_logging, get_logger

try:
    from docling.chunking import HierarchicalChunker
    from docling.document_converter import DocumentConverter
except ImportError:
    try:
        from docling_core.transforms.chunker import HierarchicalChunker
        from docling.document_converter import DocumentConverter
    except ImportError:
        HierarchicalChunker = None
        DocumentConverter = None


UUID_NAMESPACE_SOURCE = UUID("9d4c1d2b-0c37-5c65-9b9a-111111111111")
UUID_NAMESPACE_CHUNK = UUID("9d4c1d2b-0c37-5c65-9b9a-222222222222")
UUID_NAMESPACE_KNOWLEDGE = UUID("9d4c1d2b-0c37-5c65-9b9a-333333333333")
UUID_NAMESPACE_RELATION = UUID("9d4c1d2b-0c37-5c65-9b9a-444444444444")

SYSTEM_PROMPT = """You are the knowledge extraction stage of a local WikiLLM pipeline.
Return ONLY JSON matching the provided JSON Schema.

Rules:
1. Preserve technical meaning from the source. Do not invent facts.
2. Keep Korean technical prose in Korean when the source is Korean.
3. Keep globally standardized technical terms in English when appropriate.
4. Every extracted item grounded in the input must retain source_refs.
5. Relations must use only the allowed ontology relation names.
6. Never invent a source reference that is not derivable from the supplied metadata.
7. Prefer a small number of high-confidence concepts/entities/relations over noisy extraction.
8. Do not generate IDs. Identity is assigned deterministically by the pipeline.
9. For relation source/target, use the exact semantic name of the extracted concept/entity.
10. Every relation source AND target must also appear in concepts or entities in THIS response. If an endpoint is not worth extracting as a node, omit the relation.
11. Do not turn table-of-contents entries, section headings, numbered steps, page labels, or adjacent list items into relations merely because they are near each other.
12. Keep relations sparse and evidence-grounded; normally emit no more than 12 relations for one chunk.
"""


@dataclass(slots=True)
class DocumentChunk:
    id: UUID
    chunk_id: str
    source_id: UUID
    text: str
    meta: dict[str, Any]


class OntologyRelation(BaseModel):
    source: list[str]
    target: list[str]


class OntologyConfig(BaseModel):
    version: int = 1
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[str, OntologyRelation]


def load_ontology(config_path: Path) -> OntologyConfig:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return OntologyConfig.model_validate(data)


def source_id_from_bytes(data: bytes) -> UUID:
    digest = hashlib.sha256(data).hexdigest()
    return uuid5(UUID_NAMESPACE_SOURCE, digest)


def source_id(path: Path) -> UUID:
    return source_id_from_bytes(path.read_bytes())


def chunk_uuid(source_id: UUID, chunk_key: str) -> UUID:
    return uuid5(UUID_NAMESPACE_CHUNK, f"{source_id}:{chunk_key}")


def knowledge_uuid(doc_type: DocumentType, name: str) -> UUID:
    canonical = re.sub(r"\s+", " ", name.strip().casefold())
    return uuid5(UUID_NAMESPACE_KNOWLEDGE, f"{doc_type.value}:{canonical}")


def relation_uuid(source: UUID, relation: RelationType, target: UUID, evidence: str) -> UUID:
    evidence_key = re.sub(r"\s+", " ", evidence.strip().casefold())
    return uuid5(UUID_NAMESPACE_RELATION, f"{source}:{relation.value}:{target}:{evidence_key}")


def chunk_document(source: Path, max_pages: int | None = None) -> list[DocumentChunk]:
    logger = get_logger(__name__)
    logger.info("Docling START source=%s max_pages=%s", source, max_pages)
    if DocumentConverter is None or HierarchicalChunker is None:
        logger.error("E_DOCLING_DEPENDENCY: Docling imports unavailable. DocumentConverter=%s HierarchicalChunker=%s", DocumentConverter, HierarchicalChunker)
        raise RuntimeError("Docling is required for document ingestion. Install requirements.txt first.")
    converter_kwargs: dict[str, Any] = {}
    if max_pages is not None:
        converter_kwargs["max_num_pages"] = max_pages
    try:
        result = DocumentConverter(**converter_kwargs).convert(source)
        document = result.document
        logger.info("Docling conversion complete source=%s", source)
        chunker = HierarchicalChunker()
    except Exception:
        logger.exception("E_DOCLING_CONVERSION: Docling conversion failed source=%s", source)
        raise
    source_uuid = source_id(source)
    chunks: list[DocumentChunk] = []
    for idx, chunk in enumerate(chunker.chunk(document)):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        meta_obj = getattr(chunk, "meta", None)
        if meta_obj is None:
            meta: dict[str, Any] = {}
        elif hasattr(meta_obj, "export_json_dict"):
            meta = meta_obj.export_json_dict()
        elif hasattr(meta_obj, "model_dump"):
            meta = meta_obj.model_dump(mode="json")
        else:
            meta = {"repr": repr(meta_obj)}
        chunk_key = f"{source.stem}:{idx:04d}"
        chunks.append(
            DocumentChunk(
                id=chunk_uuid(source_uuid, chunk_key),
                chunk_id=chunk_key,
                source_id=source_uuid,
                text=text,
                meta=meta,
            )
        )
    logger.info(
        "Docling END source=%s chunks=%d non_empty_chars=%d",
        source,
        len(chunks),
        sum(len(c.text) for c in chunks),
    )
    return chunks


def parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    raw = text[4:end]
    data = yaml.safe_load(raw) or {}
    return data if isinstance(data, dict) else {}


def parse_index(index_path: Path) -> list[tuple[str, str]]:
    if not index_path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*[-*]\s*\[(.+?)\]\((.+?\.md)\)\s*$", line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


def retrieve_existing_context(llmwiki_root: Path, query: str, top_k: int = 5) -> str:
    pairs = parse_index(llmwiki_root / "wiki/index.md")
    if not pairs:
        return "(No existing wiki index entries.)"
    terms = {term for term in re.findall(r"[0-9A-Za-z가-힣]{2,}", query.lower())}
    scored: list[tuple[int, str, str]] = []
    for title, relative_path in pairs:
        score = sum(1 for term in terms if term in title.lower() or term in relative_path.lower())
        scored.append((score, title, relative_path))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    blocks: list[str] = []
    for _, title, relative_path in scored[:top_k]:
        full = llmwiki_root / "wiki" / relative_path
        body = full.read_text(encoding="utf-8") if full.exists() else ""
        blocks.append(f"### Existing note: {title}\nPath: wiki/{relative_path}\n{body[:4000]}")
    return "\n\n".join(blocks)


def build_prompt(*, source_path: Path, source_rel: str, chunk: DocumentChunk, existing_context: str, ontology: OntologyConfig) -> str:
    title = source_path.stem.replace("_", " ")
    allowed = sorted(ontology.relations.keys())
    return f"""Extract durable wiki knowledge from this document chunk.

SOURCE_FILE: {source_rel}
SOURCE_ID: {chunk.source_id}
CHUNK_ID: {chunk.id}
CHUNK_KEY: {chunk.chunk_id}
DOCUMENT_TITLE: {title}
ALLOWED_RELATIONS: {json.dumps(allowed, ensure_ascii=False)}

Existing wiki context is reference-only. Update/merge it conceptually; do not blindly duplicate it.
{existing_context}

DOCUMENT CHUNK METADATA:
{json.dumps(chunk.meta, ensure_ascii=False, indent=2, default=str)}

DOCUMENT CHUNK TEXT:
---
{chunk.text}
---

For every source reference, use source_file="{source_rel}", source_id="{chunk.source_id}", chunk_id="{chunk.id}".
Locator may be a heading/page when visible in metadata. Keep quote short and grounded.
"""


def citation_preserved(result: KnowledgeExtraction, source_rel: str, source_uuid: UUID, chunk_uuid_value: UUID) -> bool:
    refs: list[SourceRef] = list(result.source_refs)
    for item in result.concepts:
        refs.extend(item.source_refs)
    for item in result.entities:
        refs.extend(item.source_refs)
    for item in result.relations:
        refs.extend(item.source_refs)
    return any(
        ref.source_file == source_rel and ref.source_id == source_uuid and ref.chunk_id == chunk_uuid_value
        for ref in refs
    )


def _name_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def build_knowledge_registry(llmwiki_root: Path) -> tuple[dict[UUID, KnowledgeDocument], dict[str, tuple[UUID, DocumentType]]]:
    documents: dict[UUID, KnowledgeDocument] = {}
    names: dict[str, tuple[UUID, DocumentType]] = {}
    wiki_root = llmwiki_root / "wiki"
    if not wiki_root.exists():
        return documents, names
    for document, _ in load_knowledge_documents(wiki_root):
        documents[document.id] = document
        names[_name_key(document.title)] = (document.id, document.type)
        for alias in document.aliases:
            names.setdefault(_name_key(alias), (document.id, document.type))
    return documents, names


def resolve_type(
    name: str,
    extraction: KnowledgeExtraction,
    known_names: dict[str, tuple[UUID, DocumentType]] | None = None,
) -> tuple[UUID, DocumentType] | None:
    key = _name_key(name)
    for concept in extraction.concepts:
        if _name_key(concept.name) == key:
            return knowledge_uuid(DocumentType.CONCEPT, concept.name), DocumentType.CONCEPT
    for entity in extraction.entities:
        if _name_key(entity.name) == key:
            return knowledge_uuid(DocumentType.ENTITY, entity.name), DocumentType.ENTITY
    if known_names is not None:
        return known_names.get(key)
    return None


def materialize_relations(
    extraction: KnowledgeExtraction,
    ontology: OntologyConfig,
    *,
    known_documents: dict[UUID, KnowledgeDocument] | None = None,
    known_names: dict[str, tuple[UUID, DocumentType]] | None = None,
) -> tuple[list[KnowledgeDocument], list[str]]:
    docs: dict[UUID, KnowledgeDocument] = {}
    errors: list[str] = []
    for concept in extraction.concepts:
        uid = knowledge_uuid(DocumentType.CONCEPT, concept.name)
        docs[uid] = KnowledgeDocument(
            id=uid,
            type=DocumentType.CONCEPT,
            title=concept.name,
            aliases=concept.aliases,
            summary=concept.summary,
            definition=concept.definition,
            sources=concept.source_refs or extraction.source_refs,
        )
    for entity in extraction.entities:
        uid = knowledge_uuid(DocumentType.ENTITY, entity.name)
        docs[uid] = KnowledgeDocument(
            id=uid,
            type=DocumentType.ENTITY,
            title=entity.name,
            aliases=entity.aliases,
            summary=entity.description,
            entity_type=entity.entity_type,
            sources=entity.source_refs or extraction.source_refs,
        )

    for rel in extraction.relations:
        source_resolved = resolve_type(rel.source, extraction, known_names)
        target_resolved = resolve_type(rel.target, extraction, known_names)
        if source_resolved is None or target_resolved is None:
            errors.append(f"unresolved relation endpoint: {rel.source} --{rel.relation.value}--> {rel.target}")
            continue
        source_id_value, source_type = source_resolved
        target_id_value, target_type = target_resolved
        rule = ontology.relations.get(rel.relation.value)
        if rule is None or source_type.value not in rule.source or target_type.value not in rule.target:
            errors.append(
                f"ontology violation: {source_type.value}({rel.source}) --{rel.relation.value}--> {target_type.value}({rel.target})"
            )
            continue
        if source_id_value not in docs and known_documents and source_id_value in known_documents:
            docs[source_id_value] = known_documents[source_id_value].model_copy(deep=True)
        if source_id_value not in docs:
            errors.append(f"unresolved relation source document: {rel.source}")
            continue
        relationship = Relation(
            id=relation_uuid(source_id_value, rel.relation, target_id_value, rel.evidence),
            source=source_id_value,
            relation=rel.relation,
            target=target_id_value,
            evidence=rel.evidence,
            source_refs=rel.source_refs or extraction.source_refs,
            confidence=rel.confidence,
        )
        docs[source_id_value].relations.append(relationship)
    return list(docs.values()), errors


def serialize_frontmatter(document: KnowledgeDocument) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": str(document.id),
        "type": document.type.value,
        "title": document.title,
    }
    if document.aliases:
        data["aliases"] = document.aliases
    if document.summary:
        data["summary"] = document.summary
    if document.definition:
        data["definition"] = document.definition
    if document.entity_type:
        data["entity_type"] = document.entity_type
    data["sources"] = [
        {
            "source_id": str(ref.source_id),
            "chunk_id": str(ref.chunk_id),
            "source_file": ref.source_file,
            "locator": ref.locator,
            "quote": ref.quote,
        }
        for ref in document.sources
    ]
    data["relations"] = [
        {
            "id": str(rel.id),
            "source": str(rel.source),
            "type": rel.relation.value,
            "target": str(rel.target),
            "evidence": rel.evidence,
            "source_refs": [
                {
                    "source_id": str(ref.source_id),
                    "chunk_id": str(ref.chunk_id),
                    "source_file": ref.source_file,
                    "locator": ref.locator,
                    "quote": ref.quote,
                }
                for ref in rel.source_refs
            ],
        }
        for rel in document.relations
    ]
    return data


def render_document(document: KnowledgeDocument, title_lookup: dict[UUID, tuple[DocumentType, str, str]] | None = None) -> str:
    frontmatter = yaml.safe_dump(serialize_frontmatter(document), allow_unicode=True, sort_keys=False).strip()
    lines = ["---", frontmatter, "---", "", f"# {document.title}", ""]
    if document.summary:
        lines += [f"> {document.summary}", ""]
    if document.definition:
        lines += ["## Definition", "", document.definition, ""]
    if document.entity_type:
        lines += [f"**Type:** {document.entity_type}", ""]
    if document.aliases:
        lines += ["## Aliases", "", ", ".join(document.aliases), ""]
    if document.relations:
        lines += ["## Relations", ""]
        for rel in document.relations:
            if title_lookup and rel.target in title_lookup:
                target_type, target_title, target_path = title_lookup[rel.target]
                target_link = f"[[{target_path}|{target_title}]]"
            else:
                target_link = f"`{rel.target}`"
            suffix = f" — {rel.evidence}" if rel.evidence else ""
            lines.append(f"- `{rel.relation.value}` → {target_link}{suffix}")
        lines.append("")
    lines += ["## Sources", ""]
    seen: set[tuple[UUID, UUID]] = set()
    for ref in document.sources:
        key = (ref.source_id, ref.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        locator = f" ({ref.locator})" if ref.locator else ""
        lines.append(f"- `{ref.source_file}`{locator} — chunk `{ref.chunk_id}` — source `{ref.source_id}`")
    return "\n".join(lines) + "\n"


def _title_lookup(llmwiki_root: Path) -> dict[UUID, tuple[DocumentType, str, str]]:
    lookup: dict[UUID, tuple[DocumentType, str, str]] = {}
    for folder, document_type in (("concepts", DocumentType.CONCEPT), ("entities", DocumentType.ENTITY), ("sources", DocumentType.SOURCE), ("comparisons", DocumentType.COMPARISON)):
        folder_path = llmwiki_root / "wiki" / folder
        if not folder_path.exists():
            continue
        for path in folder_path.glob("*.md"):
            meta = parse_frontmatter(path.read_text(encoding="utf-8"))
            try:
                uid = UUID(str(meta.get("id")))
                title = str(meta.get("title", path.stem))
                rel_stem = str(path.relative_to(llmwiki_root / "wiki").with_suffix("")).replace("\\", "/")
                lookup[uid] = (document_type, title, rel_stem)
            except Exception:
                continue
    return lookup


def upsert_document(
    llmwiki_root: Path,
    document: KnowledgeDocument,
    title_lookup: dict[UUID, tuple[DocumentType, str, str]] | None = None,
) -> str:
    logger = get_logger(__name__)
    folder = {
        DocumentType.CONCEPT: "concepts",
        DocumentType.ENTITY: "entities",
        DocumentType.SOURCE: "sources",
        DocumentType.COMPARISON: "comparisons",
    }[document.type]
    wiki_root = llmwiki_root / "wiki"
    if title_lookup is None:
        title_lookup = _title_lookup(llmwiki_root)

    # UUID is the canonical identity. If this UUID was already materialized,
    # keep using the exact existing path even if the title/slug changes.
    existing_lookup = title_lookup.get(document.id)
    if existing_lookup is not None:
        _, _, existing_rel_stem = existing_lookup
        path = wiki_root / f"{existing_rel_stem}.md"
    else:
        base_slug = stable_slug(document.title)
        path = wiki_root / folder / f"{base_slug}.md"
        if path.exists():
            old_meta = parse_frontmatter(path.read_text(encoding="utf-8"))
            old_id = str(old_meta.get("id") or "")
            if old_id and old_id != str(document.id):
                # Different semantic identities can normalize to the same filename
                # (for example multiple empty/unsupported titles -> untitled.md).
                # Preserve a readable slug, but disambiguate the storage path with
                # a deterministic UUID suffix instead of aborting Wiki generation.
                path = wiki_root / folder / f"{base_slug}--{document.id.hex[:8]}.md"
                if path.exists():
                    suffix_meta = parse_frontmatter(path.read_text(encoding="utf-8"))
                    suffix_id = str(suffix_meta.get("id") or "")
                    if suffix_id and suffix_id != str(document.id):
                        path = wiki_root / folder / f"{base_slug}--{document.id}.md"
                logger.warning(
                    "WIKI slug collision resolved title=%r existing_id=%s new_id=%s path=%s",
                    document.title,
                    old_id,
                    document.id,
                    path,
                )

    rel_path = str(path.relative_to(wiki_root)).replace("\\", "/")
    rel_stem = str(path.relative_to(wiki_root).with_suffix("")).replace("\\", "/")
    title_lookup[document.id] = (document.type, document.title, rel_stem)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_document(document, title_lookup), encoding="utf-8")
        return rel_path

    old = path.read_text(encoding="utf-8")
    old_meta = parse_frontmatter(old)
    if old_meta.get("id") and str(old_meta["id"]) != str(document.id):
        raise ValueError(f"identity collision at {path}: existing id={old_meta['id']} new id={document.id}")

    existing_sources = old_meta.get("sources") or []
    existing_relation_ids = {str(item.get("id")) for item in old_meta.get("relations") or [] if isinstance(item, dict)}
    merged_sources = list(existing_sources)
    source_keys = {(str(item.get("source_id")), str(item.get("chunk_id"))) for item in existing_sources if isinstance(item, dict)}
    for ref in serialize_frontmatter(document)["sources"]:
        if (ref["source_id"], ref["chunk_id"]) not in source_keys:
            merged_sources.append(ref)

    merged_relations = list(old_meta.get("relations") or [])
    for rel in serialize_frontmatter(document)["relations"]:
        if rel["id"] not in existing_relation_ids:
            merged_relations.append(rel)

    merged = document.model_copy(update={
        "sources": [SourceRef.model_validate(item) for item in merged_sources],
        "relations": [
            Relation(
                id=UUID(str(item["id"])),
                source=UUID(str(item["source"])),
                relation=RelationType(str(item["type"])),
                target=UUID(str(item["target"])),
                evidence=str(item.get("evidence", "")),
                source_refs=[SourceRef.model_validate(ref) for ref in item.get("source_refs", []) or []],
            )
            for item in merged_relations
        ],
    })
    title_lookup[merged.id] = (merged.type, merged.title, rel_stem)
    path.write_text(render_document(merged, title_lookup), encoding="utf-8")
    return rel_path


def append_source_note(llmwiki_root: Path, source: Path, source_uuid: UUID, chunks: list[DocumentChunk]) -> str:
    document = KnowledgeDocument(id=source_uuid, type=DocumentType.SOURCE, title=source.stem, summary=str(source))
    source_note = llmwiki_root / "wiki/sources" / f"{stable_slug(source.stem)}.md"
    front = {
        "id": str(source_uuid),
        "type": "source",
        "title": source.stem,
        "source_file": str(source),
        "chunks": [{"id": str(chunk.id), "key": chunk.chunk_id} for chunk in chunks],
    }
    text = "---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n# " + source.stem + "\n\n"
    if not source_note.exists():
        source_note.parent.mkdir(parents=True, exist_ok=True)
        source_note.write_text(text, encoding="utf-8")
    return str(source_note.relative_to(llmwiki_root / "wiki"))


def upsert_index(llmwiki_root: Path, entries: list[tuple[str, str, UUID]]) -> None:
    index_path = llmwiki_root / "wiki/index.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# LLM Wiki Index\n\n"
    for title, relative_path, uid in entries:
        line = f"- [{title}]({relative_path}) <!-- id: {uid} -->"
        if str(uid) not in existing:
            existing += line + "\n"
    index_path.write_text(existing, encoding="utf-8")


def append_log(llmwiki_root: Path, message: str) -> None:
    log_path = llmwiki_root / "wiki/log.md"
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- `{timestamp}` {message}\n")


def source_ref_defaults(result: KnowledgeExtraction, chunk: DocumentChunk, source_rel: str) -> KnowledgeExtraction:
    """Materialize LLM provenance into pipeline-owned UUIDs."""
    def normalize(ref: SourceRefDraft) -> SourceRef:
        return SourceRef(
            source_id=chunk.source_id,
            chunk_id=chunk.id,
            source_file=source_rel,
            locator=ref.locator,
            quote=ref.quote,
        )

    return result.model_copy(update={
        "source_refs": [normalize(ref) for ref in result.source_refs] or [
            SourceRef(source_id=chunk.source_id, chunk_id=chunk.id, source_file=source_rel)
        ],
        "concepts": [c.model_copy(update={"source_refs": [normalize(ref) for ref in c.source_refs]}) for c in result.concepts],
        "entities": [e.model_copy(update={"source_refs": [normalize(ref) for ref in e.source_refs]}) for e in result.entities],
        "relations": [r.model_copy(update={"source_refs": [normalize(ref) for ref in r.source_refs]}) for r in result.relations],
    })


def run_ingest(*, source: Path, llmwiki_root: Path, ollama_host: str, model: str, ontology_path: Path, max_retries: int = 1, max_existing_notes: int = 5, save_chunks: bool = True) -> dict[str, Any]:
    logger = get_logger(__name__)
    logger.info(
        "Ingest START source=%s model=%s host=%s ontology=%s max_retries=%d",
        source,
        model,
        ollama_host,
        ontology_path,
        max_retries,
    )
    ensure_layout(llmwiki_root)
    if not source.exists():
        raise FileNotFoundError(source)

    try:
        ontology = load_ontology(ontology_path)
        logger.info("Ontology loaded relation_count=%d", len(ontology.relations))
        source_uuid = source_id(source)
        chunks = chunk_document(source)
    except Exception:
        logger.exception("E_INGEST_PREPARATION: Ingest preparation failed source=%s", source)
        raise
    source_rel = str(source.resolve().relative_to(llmwiki_root.resolve())) if source.resolve().is_relative_to(llmwiki_root.resolve()) else str(source.resolve())
    try:
        client = OllamaClient(host=ollama_host)
    except Exception:
        logger.exception("E_OLLAMA_INIT: Failed to initialize OllamaClient host=%s", ollama_host)
        raise

    chunk_output = APP_ROOT / "outputs" / "chunks" / source.stem
    chunk_output.mkdir(parents=True, exist_ok=True)
    if save_chunks:
        for chunk in chunks:
            record = ChunkRecord(id=chunk.id, source_id=chunk.source_id, source_file=source_rel, ordinal=int(chunk.chunk_id.rsplit(":", 1)[-1]), text=chunk.text, meta=chunk.meta)
            (chunk_output / f"{chunk.id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    existing_context = retrieve_existing_context(llmwiki_root, source.stem.replace("_", " "), top_k=max_existing_notes)
    title_lookup = _title_lookup(llmwiki_root)
    known_documents, known_names = build_knowledge_registry(llmwiki_root)
    ingest_started = time.monotonic()
    progress_every = max(1, int(os.getenv("LLMWIKI_PROGRESS_EVERY_CHUNKS", "10")))
    results: list[dict[str, Any]] = []
    index_entries: list[tuple[str, str, UUID]] = []
    errors: list[str] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        logger.info("INGEST CHUNK START %d/%d chunk_id=%s chars=%d", chunk_index, len(chunks), chunk.id, len(chunk.text))
        prompt = build_prompt(source_path=source, source_rel=source_rel, chunk=chunk, existing_context=existing_context, ontology=ontology)
        try:
            call = client.generate_structured(
                model=model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                response_model=KnowledgeExtraction,
                options={"temperature": 0.0, "num_ctx": 8192, "seed": 42},
                max_retries=max_retries,
                think=False,
                request_id=f"ingest:{chunk_index}/{len(chunks)}:{chunk.id}",
            )
        except Exception as exc:
            logger.exception("INGEST CHUNK EXCEPTION chunk_id=%s", chunk.id)
            results.append(
                {
                    "chunk_id": str(chunk.id),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if call.parsed is None:
            logger.error(
                "INGEST CHUNK FAILED chunk_id=%s attempts=%d error=%s raw_chars=%d",
                chunk.id,
                call.attempts,
                call.error,
                len(call.raw_text or ""),
            )
            results.append({"chunk_id": str(chunk.id), "success": False, "error": call.error})
            continue

        logger.info(
            "INGEST CHUNK LLM SUCCESS chunk=%d/%d chunk_id=%s attempts=%d raw_chars=%d ttft_ms=%s wall_ms=%.2f total_ms=%s load_ms=%s prompt_tokens=%s prompt_eval_ms=%s output_tokens=%s eval_ms=%s tps=%s",
            chunk_index,
            len(chunks),
            chunk.id,
            call.attempts,
            len(call.raw_text or ""),
            f"{call.ttft_ms:.2f}" if call.ttft_ms is not None else "None",
            call.wall_time_ms,
            f"{call.total_duration_ms:.2f}" if call.total_duration_ms is not None else "None",
            f"{call.load_duration_ms:.2f}" if call.load_duration_ms is not None else "None",
            call.prompt_eval_count,
            f"{call.prompt_eval_duration_ms:.2f}" if call.prompt_eval_duration_ms is not None else "None",
            call.eval_count,
            f"{call.eval_duration_ms:.2f}" if call.eval_duration_ms is not None else "None",
            f"{call.tokens_per_sec:.2f}" if call.tokens_per_sec is not None else "None",
        )

        extraction = source_ref_defaults(call.parsed, chunk, source_rel)
        try:
            documents, ontology_errors = materialize_relations(
                extraction,
                ontology,
                known_documents=known_documents,
                known_names=known_names,
            )
            errors.extend(ontology_errors)
            if ontology_errors:
                logger.warning(
                    "E_ONTOLOGY_VIOLATION: INGEST ONTOLOGY WARN chunk_id=%s errors=%s",
                    chunk.id,
                    ontology_errors[:10],
                )
            for document in documents:
                rel_path = upsert_document(llmwiki_root, document, title_lookup=title_lookup)
                logger.debug(
                    "WIKI WRITE chunk_id=%s document=%s path=%s",
                    chunk.id,
                    document.title,
                    rel_path,
                )
                index_entries.append((document.title, rel_path, document.id))
                merged_path = llmwiki_root / "wiki" / rel_path
                try:
                    merged_document = document_from_markdown(merged_path)
                    known_documents[merged_document.id] = merged_document
                    known_names[_name_key(merged_document.title)] = (merged_document.id, merged_document.type)
                    for alias in merged_document.aliases:
                        known_names.setdefault(_name_key(alias), (merged_document.id, merged_document.type))
                except Exception:
                    logger.exception("E_WIKI_REGISTRY_REFRESH: path=%s", merged_path)
        except Exception:
            logger.exception("E_WIKI_WRITE: INGEST MATERIALIZATION/WIKI WRITE FAILED chunk_id=%s", chunk.id)
            raise

        results.append({
            "chunk_id": str(chunk.id),
            "success": True,
            "schema_valid": call.schema_valid,
            "recovered_json": call.recovered_json,
            "ttft_ms": call.ttft_ms,
            "tokens_per_sec": call.tokens_per_sec,
            "ontology_compliance_rate": (
                1.0
                if not extraction.relations
                else max(0.0, (len(extraction.relations) - len(ontology_errors)) / len(extraction.relations))
            ),
            "source_citation_preserved": citation_preserved(extraction, source_rel, source_uuid, chunk.id),
        })
        logger.info("INGEST CHUNK END chunk_id=%s", chunk.id)
        if chunk_index % progress_every == 0 or chunk_index == len(chunks):
            elapsed = max(time.monotonic() - ingest_started, 0.001)
            rate = chunk_index / elapsed
            eta_s = (len(chunks) - chunk_index) / rate if rate > 0 else None
            logger.info(
                "INGEST PROGRESS completed=%d/%d progress=%.2f%% rate_chunks_min=%.2f elapsed_s=%.1f eta_s=%s",
                chunk_index,
                len(chunks),
                chunk_index / len(chunks) * 100.0,
                rate * 60.0,
                elapsed,
                f"{eta_s:.1f}" if eta_s is not None else "None",
            )

    source_note = append_source_note(llmwiki_root, source, source_uuid, chunks)
    upsert_index(llmwiki_root, index_entries + [(source.stem, source_note, source_uuid)])
    append_log(llmwiki_root, f"ingested `{source_rel}` with `{model}`: {sum(1 for x in results if x.get('success'))}/{len(chunks)} chunks succeeded; source_id={source_uuid}")
    logger.info(
        "Ingest END source=%s success_chunks=%d/%d ontology_errors=%d",
        source_rel,
        sum(1 for x in results if x.get("success")),
        len(chunks),
        len(errors),
    )

    write_index(llmwiki_root / "wiki")
    return {"source": source_rel, "source_id": str(source_uuid), "model": model, "chunks": len(chunks), "results": results, "ontology_errors": errors}


def main() -> None:
    log_path = os.getenv("LLMWIKI_LOG_FILE")
    configure_logging(
        component=__name__,
        log_file=Path(log_path) if log_path else APP_ROOT / "outputs" / "logs" / "ingest.log",
    )
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Docling -> Ollama -> UUID-backed Obsidian Wiki ingest pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--llmwiki", default=str(DEFAULT_LLMWIKI_ROOT))
    parser.add_argument("--model", default="qwen3:4b-q4_K_M")
    parser.add_argument(
    "--ollama-host",
    default=os.getenv(
        "OLLAMA_HOST",
        "http://localhost:3003/api/proxy",
    ),
)
    parser.add_argument("--ontology", default=str(APP_ROOT.parent / "config" / "ontology.yaml"))
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run_ingest(
            source=Path(args.input).expanduser().resolve(),
            llmwiki_root=resolve_root(args.llmwiki, DEFAULT_LLMWIKI_ROOT),
            ollama_host=args.ollama_host,
            model=args.model,
            ontology_path=Path(args.ontology).resolve(),
            max_retries=args.max_retries,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("FATAL ingest failure")
        raise


if __name__ == "__main__":
    main()
