from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID

import yaml

from app.models.schema import RelationType
from app.pipeline.ingest import OntologyConfig, load_ontology
from app.pipeline.markdown import load_knowledge_documents


def lint(wiki_root: Path, ontology_path: Path) -> list[str]:
    problems: list[str] = []
    documents = list(load_knowledge_documents(wiki_root))
    ids = {document.id for document, _ in documents}
    relation_ids: set[UUID] = set()
    ontology = load_ontology(ontology_path)

    for document, relative in documents:
        # Source pages identify raw material and do not require another source entry.
        if document.type.value != "source" and not document.sources:
            problems.append(f"missing source provenance: {relative}")
        for relation in document.relations:
            if relation.id in relation_ids:
                problems.append(f"duplicate relation id: {relation.id}")
            relation_ids.add(relation.id)
            if relation.source != document.id:
                problems.append(f"relation source mismatch: {relative} -> {relation.source}")
            if relation.target not in ids:
                problems.append(f"missing relation target node: {relative} -> {relation.target}")
            rule = ontology.relations.get(relation.relation.value)
            if rule is None:
                problems.append(f"unknown ontology relation: {relation.relation.value} ({relative})")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint the local Markdown WikiLLM")
    parser.add_argument("--wiki", default="./llmwiki/wiki")
    parser.add_argument("--ontology", default="./config/ontology.yaml")
    args = parser.parse_args()
    problems = lint(Path(args.wiki).resolve(), Path(args.ontology).resolve())
    if problems:
        print("LLM Wiki lint: FAIL")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("LLM Wiki lint: PASS")


if __name__ == "__main__":
    main()
