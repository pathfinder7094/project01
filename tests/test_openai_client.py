from types import SimpleNamespace

from pydantic import BaseModel

import app.openai_client.client as mod


class Example(BaseModel):
    value: str


class FakeResponses:
    def parse(self, **kwargs):
        assert kwargs["model"] == "gpt-test"
        assert kwargs["store"] is False
        assert kwargs["text_format"] is Example
        return SimpleNamespace(
            output_text='{"value":"ok"}',
            output_parsed=Example(value="ok"),
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        )


class FakeSDK:
    def __init__(self, **kwargs):
        assert kwargs["api_key"] == "secret"
        assert kwargs["max_retries"] == 0
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        self.responses = FakeResponses()


def test_openai_client_resolves_named_environment(monkeypatch):
    monkeypatch.setenv("MY_OPENAI_KEY", "secret")
    monkeypatch.setattr(mod, "OpenAI", FakeSDK)
    client = mod.OpenAIClient(api_key_env="MY_OPENAI_KEY")
    result = client.generate_structured(
        model="gpt-test",
        system="system",
        prompt="prompt",
        response_model=Example,
        max_retries=0,
    )
    assert result.schema_valid is True
    assert result.parsed == Example(value="ok")
    assert result.prompt_eval_count == 12
    assert result.eval_count == 4
    assert result.tokens_per_sec is None
