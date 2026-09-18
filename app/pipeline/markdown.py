from __future__ import annotations

from pathlib import Path
from uuid import UUID

import yaml

from app.models.schema import KnowledgeDocument, Relation, RelationType, SourceRef


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    data = yaml.safe_load(text[4:end]) or {}
    return data if isinstance(data, dict) else {}


def document_from_markdown(path: Path) -> KnowledgeDocument:
    meta = parse_frontmatter(path.read_text(encoding="utf-8"))
    relations = []
    for item in meta.get("relations", []) or []:
        relations.append(
            Relation(
                id=UUID(str(item["id"])),
                source=UUID(str(item["source"])),
                relation=RelationType(str(item["type"])),
                target=UUID(str(item["target"])),
                evidence=str(item.get("evidence", "")),
                source_refs=[SourceRef.model_validate(ref) for ref in item.get("source_refs", []) or []],
            )
        )
    return KnowledgeDocument(
        id=UUID(str(meta["id"])),
        type=meta["type"],
        title=str(meta["title"]),
        aliases=list(meta.get("aliases", []) or []),
        summary=str(meta.get("summary", "")),
        definition=str(meta.get("definition", "")),
        entity_type=meta.get("entity_type"),
        sources=[SourceRef.model_validate(ref) for ref in meta.get("sources", []) or []],
        relations=relations,
    )


def load_knowledge_documents(wiki_root: Path):
    for path in sorted(wiki_root.rglob("*.md")):
        if path.name in {"index.md", "log.md", "overview.md"}:
            continue
        try:
            document = document_from_markdown(path)
        except Exception:
            continue
        yield document, str(path.relative_to(wiki_root)).replace("\\", "/")


def render_wikilinks(document: KnowledgeDocument, title_lookup: dict[UUID, str] | None = None) -> list[str]:
    lines: list[str] = []
    for relation in document.relations:
        target = title_lookup.get(relation.target, str(relation.target)) if title_lookup else str(relation.target)
        lines.append(f"- `{relation.relation.value}` → [[{target}]]")
    return lines


def write_index(wiki_root: Path) -> None:
    docs = sorted(load_knowledge_documents(wiki_root), key=lambda item: (item[0].type.value, item[0].title.casefold()))
    lines = ["# LLM Wiki Index", "", "## Contents", ""]
    current = None
    for document, rel in docs:
        folder = document.type.value
        if folder != current:
            current = folder
            lines.extend([f"### {folder}", ""])
        display_path = rel.replace(" ", "%20")
        lines.append(f"- [{document.title}]({display_path}) <!-- id: {document.id} -->")
    (wiki_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
