from pathlib import Path

from app.eval.batch_benchmark import read_openai_model_list


def test_openai_model_mapping(tmp_path: Path):
    path = tmp_path / "models.txt"
    path.write_text(
        "# model:key env\n"
        "gpt-5.6-luna:OPENAI_API_KEY\n"
        "gpt-5.6-terra:OPENAI_API_KEY_TEAM\n",
        encoding="utf-8",
    )
    targets = read_openai_model_list(path)
    assert [(t.provider, t.model, t.api_key_env) for t in targets] == [
        ("openai", "gpt-5.6-luna", "OPENAI_API_KEY"),
        ("openai", "gpt-5.6-terra", "OPENAI_API_KEY_TEAM"),
    ]


def test_openai_model_mapping_rejects_missing_env(tmp_path: Path):
    path = tmp_path / "models.txt"
    path.write_text("gpt-5.6-luna\n", encoding="utf-8")
    try:
        read_openai_model_list(path)
    except ValueError as exc:
        assert "MODEL:API_KEY_ENV" in str(exc)
    else:
        raise AssertionError("expected ValueError")
