from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeVar

try:
    import ollama
except ImportError:  # optional until a real Ollama call is made
    ollama = None
from pydantic import BaseModel, ValidationError

from app.utils.logging import get_logger

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class StructuredCallResult:
    parsed: BaseModel | None
    raw_text: str
    schema_valid: bool
    recovered_json: bool
    attempts: int
    ttft_ms: float | None
    wall_time_ms: float
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_count: int | None
    prompt_eval_duration_ms: float | None
    eval_count: int | None
    eval_duration_ms: float | None
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float | None:
        if self.eval_count is None or not self.eval_duration_ms or self.eval_duration_ms <= 0:
            return None
        return self.eval_count / (self.eval_duration_ms / 1000.0)


class OllamaClient:
    """Small synchronous wrapper around Ollama's structured-output chat API.

    The client intentionally stays synchronous. Bounded benchmark concurrency is handled
    by the benchmark runner so one call remains easy to reason about and retry.
    """

    def __init__(self, host: str | None = None) -> None:
        self.logger = get_logger(__name__)
        self.logger.debug("Initializing OllamaClient host=%s", host)

        if ollama is None:
            self.logger.error("E_OLLAMA_DEPENDENCY: python ollama package is not installed")
            raise RuntimeError(
                "The ollama Python package is required for OllamaClient. "
                "Install requirements.txt first."
            )

        self.host = (
            host
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:3003/api/proxy"
        ).rstrip("/")
        self.logger.info("Ollama client host=%s", self.host)

        self.api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        self.logger.info("Ollama API key present=%s", bool(self.api_key))
        if not self.api_key:
            self.logger.error("E_CONFIG_API_KEY_MISSING: OLLAMA_API_KEY is missing")
            raise RuntimeError(
                "OLLAMA_API_KEY is not set. "
                "Set it in the shell/environment; do not store secrets in config/loop.env."
            )

        self.client = ollama.Client(
            host=self.host,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.logger.info("Ollama client initialized successfully host=%s", self.host)

    def is_available(self) -> bool:
        try:
            self.client.list()
            self.logger.debug("Ollama availability check succeeded")
            return True
        except Exception:
            self.logger.exception("Ollama availability check failed")
            return False

    def list_models(self) -> list[str]:
        self.logger.debug("Listing models through host=%s", self.host)
        try:
            response = self.client.list()
        except Exception:
            self.logger.exception("Ollama list_models() failed")
            raise
        models = getattr(response, "models", None) or []
        names: list[str] = []
        for model in models:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(str(name))
        self.logger.info("Ollama models returned count=%d", len(names))
        self.logger.debug("Ollama models=%s", names)
        return names

    @staticmethod
    def _field(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    @staticmethod
    def _extract_content(response: Any) -> str:
        message = OllamaClient._field(response, "message", {})
        return str(OllamaClient._field(message, "content", "") or "")

    @staticmethod
    def _extract_json_candidate(text: str) -> str | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start >= 0 and end > start:
                candidate = text[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(0.0, value)

    def _start_watchdog(
        self,
        *,
        model: str,
        request_id: str,
        started_ns: int,
        state: dict[str, Any],
        stop_event: threading.Event,
    ) -> threading.Thread | None:
        heartbeat_seconds = self._env_float("LLMWIKI_STREAM_HEARTBEAT_SECONDS", 10.0)
        if heartbeat_seconds <= 0:
            return None
        stall_warn_seconds = self._env_float("LLMWIKI_STALL_WARN_SECONDS", 30.0)

        def _watch() -> None:
            while not stop_event.wait(heartbeat_seconds):
                now_ns = time.perf_counter_ns()
                elapsed_s = (now_ns - started_ns) / 1_000_000_000
                idle_s = (now_ns - int(state["last_activity_ns"])) / 1_000_000_000
                status = "WAITING_FIRST_TOKEN" if state["first_content_ns"] is None else "STREAMING"
                log_fn = self.logger.warning if stall_warn_seconds > 0 and idle_s >= stall_warn_seconds else self.logger.info
                log_fn(
                    "OLLAMA HEARTBEAT model=%s request_id=%s status=%s elapsed_s=%.1f idle_s=%.1f stream_chunks=%d raw_chars=%d",
                    model,
                    request_id,
                    status,
                    elapsed_s,
                    idle_s,
                    int(state["chunk_count"]),
                    int(state["raw_chars"]),
                )

        thread = threading.Thread(target=_watch, name=f"ollama-watchdog-{request_id}", daemon=True)
        thread.start()
        return thread

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
        schema = response_model.model_json_schema()
        last_error: str | None = None
        request_id = request_id or "-"

        for attempt in range(1, max_retries + 2):
            self.logger.info(
                "Ollama call start model=%s request_id=%s attempt=%d/%d prompt_chars=%d options=%s think=%s",
                model,
                request_id,
                attempt,
                max_retries + 1,
                len(prompt),
                options or {},
                think,
            )
            retry_hint = ""
            if attempt > 1:
                retry_hint = (
                    "\nPrevious output did not validate. Return ONLY a JSON object matching the schema. "
                    "Do not add markdown fences, comments, or explanatory text."
                )

            started = time.perf_counter_ns()
            first_content_ns: int | None = None
            parts: list[str] = []
            final_response: Any = None
            stop_event = threading.Event()
            state: dict[str, Any] = {
                "last_activity_ns": started,
                "first_content_ns": None,
                "chunk_count": 0,
                "raw_chars": 0,
            }
            watchdog = self._start_watchdog(
                model=model,
                request_id=request_id,
                started_ns=started,
                state=state,
                stop_event=stop_event,
            )

            try:
                chat_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt + retry_hint},
                    ],
                    "stream": True,
                    "format": schema,
                    "options": options or {},
                    "think": think,
                }
                if os.getenv("LLMWIKI_FORCE_KEEP_ALIVE_ZERO", "0").strip().lower() in {"1", "true", "yes", "on"}:
                    chat_kwargs["keep_alive"] = 0

                self.logger.debug(
                    "Calling ollama.Client.chat model=%s request_id=%s host=%s keep_alive=%r format=schema",
                    model,
                    request_id,
                    self.host,
                    chat_kwargs.get("keep_alive"),
                )
                stream = self.client.chat(**chat_kwargs)
                chunk_count = 0
                raw_chars = 0
                for chunk in stream:
                    chunk_count += 1
                    final_response = chunk
                    state["chunk_count"] = chunk_count
                    state["last_activity_ns"] = time.perf_counter_ns()
                    content = self._extract_content(chunk)
                    if content:
                        if first_content_ns is None:
                            first_content_ns = time.perf_counter_ns()
                            state["first_content_ns"] = first_content_ns
                        parts.append(content)
                        raw_chars += len(content)
                        state["raw_chars"] = raw_chars

                wall_time_ms = (time.perf_counter_ns() - started) / 1_000_000
                raw_text = "".join(parts).strip()
                candidate = self._extract_json_candidate(raw_text)
                recovered = candidate is not None and candidate != raw_text

                total_duration_ms = self._ns_to_ms(self._field(final_response, "total_duration"))
                load_duration_ms = self._ns_to_ms(self._field(final_response, "load_duration"))
                prompt_eval_count = self._field(final_response, "prompt_eval_count")
                prompt_eval_duration_ms = self._ns_to_ms(self._field(final_response, "prompt_eval_duration"))
                eval_count = self._field(final_response, "eval_count")
                eval_duration_ms = self._ns_to_ms(self._field(final_response, "eval_duration"))
                ttft_ms = ((first_content_ns - started) / 1_000_000 if first_content_ns else None)

                self.logger.info(
                    "Ollama call stream complete model=%s request_id=%s chunks=%d raw_chars=%d wall_ms=%.2f ttft_ms=%s load_ms=%s prompt_tokens=%s prompt_eval_ms=%s output_tokens=%s eval_ms=%s",
                    model,
                    request_id,
                    chunk_count,
                    len(raw_text),
                    wall_time_ms,
                    f"{ttft_ms:.2f}" if ttft_ms is not None else "None",
                    f"{load_duration_ms:.2f}" if load_duration_ms is not None else "None",
                    prompt_eval_count,
                    f"{prompt_eval_duration_ms:.2f}" if prompt_eval_duration_ms is not None else "None",
                    eval_count,
                    f"{eval_duration_ms:.2f}" if eval_duration_ms is not None else "None",
                )

                if candidate is None:
                    raise ValueError(
                        f"E_RESPONSE_JSON: response is not valid JSON "
                        f"(raw_chars={len(raw_text)}, preview={raw_text[:500]!r})"
                    )

                parsed = response_model.model_validate_json(candidate)
                result = StructuredCallResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    schema_valid=True,
                    recovered_json=recovered,
                    attempts=attempt,
                    ttft_ms=ttft_ms,
                    wall_time_ms=wall_time_ms,
                    total_duration_ms=total_duration_ms,
                    load_duration_ms=load_duration_ms,
                    prompt_eval_count=prompt_eval_count,
                    prompt_eval_duration_ms=prompt_eval_duration_ms,
                    eval_count=eval_count,
                    eval_duration_ms=eval_duration_ms,
                )
                self.logger.info(
                    "Ollama call success model=%s request_id=%s attempt=%d schema_valid=true recovered_json=%s tokens_per_sec=%s",
                    model,
                    request_id,
                    attempt,
                    recovered,
                    f"{result.tokens_per_sec:.2f}" if result.tokens_per_sec is not None else "None",
                )
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error_code = "E_RESPONSE_SCHEMA" if isinstance(exc, ValidationError) else "E_RESPONSE_JSON"
                last_error = f"{error_code}: {type(exc).__name__}: {exc}"
                self.logger.warning(
                    "%s model=%s request_id=%s attempt=%d/%d error=%s",
                    error_code,
                    model,
                    request_id,
                    attempt,
                    max_retries + 1,
                    last_error,
                )
                if attempt <= max_retries:
                    self.logger.info("Retrying model=%s request_id=%s next_attempt=%d", model, request_id, attempt + 1)
                    continue
                wall_time_ms = (time.perf_counter_ns() - started) / 1_000_000
                return StructuredCallResult(
                    parsed=None,
                    raw_text="".join(parts).strip(),
                    schema_valid=False,
                    recovered_json=False,
                    attempts=attempt,
                    ttft_ms=((first_content_ns - started) / 1_000_000 if first_content_ns else None),
                    wall_time_ms=wall_time_ms,
                    total_duration_ms=self._ns_to_ms(self._field(final_response, "total_duration")),
                    load_duration_ms=self._ns_to_ms(self._field(final_response, "load_duration")),
                    prompt_eval_count=self._field(final_response, "prompt_eval_count"),
                    prompt_eval_duration_ms=self._ns_to_ms(self._field(final_response, "prompt_eval_duration")),
                    eval_count=self._field(final_response, "eval_count"),
                    eval_duration_ms=self._ns_to_ms(self._field(final_response, "eval_duration")),
                    error=last_error,
                )
            except Exception as exc:  # transport/runtime errors
                last_error = f"E_OLLAMA_TRANSPORT: {type(exc).__name__}: {exc}"
                wall_time_ms = (time.perf_counter_ns() - started) / 1_000_000
                self.logger.exception(
                    "E_OLLAMA_TRANSPORT: model=%s request_id=%s attempt=%d/%d host=%s elapsed_ms=%.2f",
                    model,
                    request_id,
                    attempt,
                    max_retries + 1,
                    self.host,
                    wall_time_ms,
                )
                return StructuredCallResult(
                    parsed=None,
                    raw_text="".join(parts).strip(),
                    schema_valid=False,
                    recovered_json=False,
                    attempts=attempt,
                    ttft_ms=((first_content_ns - started) / 1_000_000 if first_content_ns else None),
                    wall_time_ms=wall_time_ms,
                    total_duration_ms=self._ns_to_ms(self._field(final_response, "total_duration")),
                    load_duration_ms=self._ns_to_ms(self._field(final_response, "load_duration")),
                    prompt_eval_count=self._field(final_response, "prompt_eval_count"),
                    prompt_eval_duration_ms=self._ns_to_ms(self._field(final_response, "prompt_eval_duration")),
                    eval_count=self._field(final_response, "eval_count"),
                    eval_duration_ms=self._ns_to_ms(self._field(final_response, "eval_duration")),
                    error=last_error,
                )
            finally:
                stop_event.set()
                if watchdog is not None:
                    watchdog.join(timeout=0.2)

        raise RuntimeError("unreachable")

    @staticmethod
    def _ns_to_ms(value: Any) -> float | None:
        return None if value is None else float(value) / 1_000_000
