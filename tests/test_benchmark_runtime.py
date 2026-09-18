from pathlib import Path
from uuid import uuid4

from app.eval.benchmark import _append_checkpoint, _ensure_checkpoint_meta, _load_checkpoint
from app.models.schema import BenchmarkMetrics


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    source_uuid = uuid4()
    chunk_uuid = uuid4()
    checkpoint = tmp_path / "model.jsonl"
    _ensure_checkpoint_meta(checkpoint, model="test-model", source_uuid=source_uuid, num_ctx=8192)
    metric = BenchmarkMetrics(
        model="test-model",
        chunk_id=chunk_uuid,
        success=True,
        schema_valid=True,
        total_duration_ms=123.4,
    )
    _append_checkpoint(checkpoint, metric)

    restored = _load_checkpoint(
        checkpoint,
        model="test-model",
        source_uuid=source_uuid,
        num_ctx=8192,
    )
    assert restored[chunk_uuid].success is True
    assert restored[chunk_uuid].total_duration_ms == 123.4


def test_incompatible_checkpoint_is_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.jsonl"
    source_uuid = uuid4()
    _ensure_checkpoint_meta(checkpoint, model="test-model", source_uuid=source_uuid, num_ctx=8192)

    restored = _load_checkpoint(
        checkpoint,
        model="other-model",
        source_uuid=source_uuid,
        num_ctx=8192,
    )
    assert restored == {}


def test_checkpoint_runtime_change_is_incompatible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.jsonl"
    source_uuid = uuid4()
    _ensure_checkpoint_meta(
        checkpoint,
        model="test-model",
        source_uuid=source_uuid,
        num_ctx=8192,
        concurrency=1,
    )

    restored = _load_checkpoint(
        checkpoint,
        model="test-model",
        source_uuid=source_uuid,
        num_ctx=8192,
        concurrency=2,
    )
    assert restored == {}
