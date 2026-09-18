from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.eval.batch_benchmark import _source_uuid, find_latest_resumable_run, slugify
from app.models.schema import DocumentType, KnowledgeDocument, Relation, RelationType
from app.pipeline.ingest import _title_lookup, upsert_document
from app.utils.paths import ensure_layout


def test_slug_collision_uses_uuid_suffix_and_preserves_wikilink(tmp_path: Path) -> None:
    root = tmp_path / "llmwiki"
    ensure_layout(root)
    lookup = _title_lookup(root)

    first = KnowledgeDocument(id=uuid4(), type=DocumentType.ENTITY, title="!!!")
    second = KnowledgeDocument(id=uuid4(), type=DocumentType.ENTITY, title="@@@")

    first_path = upsert_document(root, first, title_lookup=lookup)
    second_path = upsert_document(root, second, title_lookup=lookup)

    assert first_path == "entities/untitled.md"
    assert second_path == f"entities/untitled--{second.id.hex[:8]}.md"

    source = KnowledgeDocument(
        id=uuid4(),
        type=DocumentType.CONCEPT,
        title="Source Concept",
        relations=[
            Relation(
                id=uuid4(),
                source=uuid4(),
                relation=RelationType.RELATED_TO,
                target=second.id,
            )
        ],
    )
    source_path = upsert_document(root, source, title_lookup=lookup)
    text = (root / "wiki" / source_path).read_text(encoding="utf-8")
    assert f"[[entities/untitled--{second.id.hex[:8]}|@@@]]" in text


def test_find_latest_resumable_run_matches_source_and_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"same source bytes")
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "20260917T091553Z"
    run_dir.mkdir(parents=True)

    model = "qwen3:4b-instruct-2507-q4_K_M"
    (run_dir / "chunks.jsonl").write_text(
        json.dumps({
            "_type": "meta",
            "version": 1,
            "source_id": _source_uuid(source),
            "source_file": str(source),
            "chunks": 1,
        }) + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / f"01-{slugify(model)}.checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / f"{slugify(model)}.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / f"{slugify(model)}.extractions.jsonl").write_text("{}\n", encoding="utf-8")

    assert find_latest_resumable_run(output_dir, source, [model]) == run_dir


def test_provider_sequence_shift_reuses_legacy_ollama_checkpoint(tmp_path: Path) -> None:
    from app.eval.batch_benchmark import ModelTarget, resolve_checkpoint_dir

    run_dir = tmp_path / "20260917T091553Z"
    run_dir.mkdir()
    model = "qwen3:4b-instruct-2507-q4_K_M"
    legacy = run_dir / f"01-{slugify(model)}.checkpoints"
    legacy.mkdir()
    (legacy / f"{slugify(model)}.jsonl").write_text("{}\n", encoding="utf-8")
    (legacy / f"{slugify(model)}.extractions.jsonl").write_text("{}\n", encoding="utf-8")

    # OpenAI may now occupy sequence 1, so this Ollama target is sequence 2.
    target = ModelTarget(provider="ollama", model=model)
    assert resolve_checkpoint_dir(run_dir, 2, target) == legacy
