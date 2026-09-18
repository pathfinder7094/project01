from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

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


class StructuredLLMClient(Protocol):
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
    ) -> StructuredCallResult: ...
