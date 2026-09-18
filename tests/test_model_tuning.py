from pathlib import Path

import pytest

from app.eval.batch_benchmark import read_ollama_targets


def test_read_ollama_targets_backward_compatible(tmp_path: Path) -> None:
    path = tmp_path / "models.txt"
    path.write_text("qwen3:4b\n", encoding="utf-8")
    targets = read_ollama_targets(path)
    assert len(targets) == 1
    assert targets[0].model == "qwen3:4b"
    assert targets[0].concurrency is None
    assert targets[0].num_ctx is None


def test_read_ollama_targets_per_model_tuning(tmp_path: Path) -> None:
    path = tmp_path / "models.txt"
    path.write_text(
        "gemma3:4b-it-q4_K_M | concurrency=2 | num_ctx=8192 | num_batch=512 | num_predict=2048\n",
        encoding="utf-8",
    )
    target = read_ollama_targets(path)[0]
    assert target.model == "gemma3:4b-it-q4_K_M"
    assert target.concurrency == 2
    assert target.num_ctx == 8192
    assert target.num_batch == 512
    assert target.num_predict == 2048


def test_read_ollama_targets_rejects_unknown_tuning(tmp_path: Path) -> None:
    path = tmp_path / "models.txt"
    path.write_text("gemma3:4b | magic=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Ollama tuning key"):
        read_ollama_targets(path)
