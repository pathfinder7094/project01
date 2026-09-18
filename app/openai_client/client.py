from __future__ import annotations

import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

try:
    from openai import OpenAI
except ImportError:  # optional until an OpenAI model is configured
    OpenAI = None

from app.llm.base import StructuredCallResult
from app.utils.logging import get_logger

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Synchronous OpenAI Responses API wrapper with Pydantic Structured Outputs.

    API keys are resolved indirectly from an environment-variable name so the
    model list never contains the secret itself.  ``base_url`` is optional;
    when omitted the OpenAI SDK uses its official default endpoint.
    """

    def __init__(self, *, api_key_env: str, base_url: str | None = None) -> None:
        self.logger = get_logger(__name__)
        if OpenAI is None:
            raise RuntimeError(
                "The openai Python package is required for OpenAIClient. "
                "Run `uv sync` after updating dependencies."
            )

        self.api_key_env = api_key_env.strip()
        if not self.api_key_env:
            raise RuntimeError("OpenAI api_key_env must not be empty")
        api_key = os.getenv(self.api_key_env, "").strip()
        self.logger.info(
            "OpenAI client api_key_env=%s api_key_present=%s base_url=%s",
            self.api_key_env,
            bool(api_key),
            base_url or "<sdk-default>",
        )
        if not api_key:
            raise RuntimeError(
                f"E_CONFIG_OPENAI_API_KEY_MISSING: environment variable {self.api_key_env!r} is not set"
            )

        effective_base_url = (
            (base_url or "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or "https://api.openai.com/v1"
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": effective_base_url,
            # Benchmark retry accounting is handled by this wrapper.
            "max_retries": 0,
        }
        self.client = OpenAI(**kwargs)
        self.base_url = effective_base_url

    @staticmethod
    def _usage_field(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None) if usage is not None else None
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        response_model: type[T],
        options: dict[str, Any] | None = None,
        max_retries: int = 1,
        think: bool = False,
        request_id: str | None = None,
    ) -> StructuredCallResult:
        del options, think  # Ollama-specific knobs; intentionally ignored here.
        attempts_allowed = max(1, int(max_retries) + 1)
        last_error: str | None = None
        raw_text = ""
        overall_started = time.perf_counter_ns()

        for attempt in range(1, attempts_allowed + 1):
            started = time.perf_counter_ns()
            try:
                self.logger.info(
                    "OpenAI call start model=%s request_id=%s attempt=%d/%d",
                    model,
                    request_id or "-",
                    attempt,
                    attempts_allowed,
                )
                # responses.parse gives us the same Pydantic contract used by Ollama.
                # store=False keeps benchmark response storage disabled at the API layer.
                response = self.client.responses.parse(
                    model=model,
                    instructions=system,
                    input=prompt,
                    text_format=response_model,
                    store=False,
                )
                raw_text = str(getattr(response, "output_text", "") or "")
                parsed = getattr(response, "output_parsed", None)
                if parsed is None and raw_text:
                    parsed = response_model.model_validate_json(raw_text)
                if parsed is not None and not isinstance(parsed, response_model):
                    parsed = response_model.model_validate(parsed)

                wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                usage = getattr(response, "usage", None)
                input_tokens = self._usage_field(usage, "input_tokens")
                output_tokens = self._usage_field(usage, "output_tokens")
                self.logger.info(
                    "OpenAI call complete model=%s request_id=%s attempt=%d wall_ms=%.2f input_tokens=%s output_tokens=%s",
                    model,
                    request_id or "-",
                    attempt,
                    wall_ms,
                    input_tokens,
                    output_tokens,
                )
                return StructuredCallResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    schema_valid=parsed is not None,
                    recovered_json=False,
                    attempts=attempt,
                    ttft_ms=None,
                    wall_time_ms=wall_ms,
                    total_duration_ms=wall_ms,
                    load_duration_ms=None,
                    prompt_eval_count=input_tokens,
                    prompt_eval_duration_ms=None,
                    eval_count=output_tokens,
                    eval_duration_ms=None,
                    error=None,
                )
            except ValidationError as exc:
                last_error = f"ValidationError: {exc}"
                self.logger.warning(
                    "OpenAI structured output validation failed model=%s request_id=%s attempt=%d error=%s",
                    model,
                    request_id or "-",
                    attempt,
                    exc,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "OpenAI request failed model=%s request_id=%s attempt=%d/%d error=%s",
                    model,
                    request_id or "-",
                    attempt,
                    attempts_allowed,
                    last_error,
                )

        wall_ms = (time.perf_counter_ns() - overall_started) / 1_000_000.0
        return StructuredCallResult(
            parsed=None,
            raw_text=raw_text,
            schema_valid=False,
            recovered_json=False,
            attempts=attempts_allowed,
            ttft_ms=None,
            wall_time_ms=wall_ms,
            total_duration_ms=wall_ms,
            load_duration_ms=None,
            prompt_eval_count=None,
            prompt_eval_duration_ms=None,
            eval_count=None,
            eval_duration_ms=None,
            error=last_error or "OpenAI request failed",
        )
