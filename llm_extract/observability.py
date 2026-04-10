"""
observability.py — Per-call and session-level observability for llm-extract.

Records every attempt, raw response, validation failures, token counts,
and latency so you know exactly why structured extraction failed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import ValidationFailure


# ---------------------------------------------------------------------------
# Per-attempt record
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    attempt_number: int
    provider: str
    model: str
    raw_response: str
    parsed_data: Optional[Dict[str, Any]]
    validation_failures: List[ValidationFailure]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    succeeded: bool
    error: Optional[str] = None  # exception message if the SDK call itself failed


# ---------------------------------------------------------------------------
# ForgeReport — result of observer.report()
# ---------------------------------------------------------------------------

@dataclass
class ForgeReport:
    """Complete observability report for a single forge() call."""

    # High-level
    succeeded: bool
    total_attempts: int
    winning_attempt: Optional[int]  # 1-indexed; None if all failed
    provider: str
    model: str

    # Per-attempt detail
    attempts: List[AttemptRecord] = field(default_factory=list)

    # Aggregates
    @property
    def total_input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def total_output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def validation_failures(self) -> List[ValidationFailure]:
        """All failures from all attempts (flattened)."""
        all_failures: List[ValidationFailure] = []
        for attempt in self.attempts:
            all_failures.extend(attempt.validation_failures)
        return all_failures

    @property
    def raw_responses(self) -> List[str]:
        return [a.raw_response for a in self.attempts]

    @property
    def latency_ms(self) -> List[float]:
        return [a.latency_ms for a in self.attempts]

    @property
    def tokens_used(self) -> Dict[str, int]:
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
            "total": self.total_tokens,
        }

    def summary(self) -> str:
        lines = [
            f"ForgeReport(succeeded={self.succeeded}, attempts={self.total_attempts})",
            f"  provider={self.provider!r}, model={self.model!r}",
            f"  tokens: input={self.total_input_tokens}, output={self.total_output_tokens}",
            f"  latency: {self.total_latency_ms:.1f}ms total",
        ]
        for i, attempt in enumerate(self.attempts, 1):
            status = "OK" if attempt.succeeded else f"FAIL({len(attempt.validation_failures)} errors)"
            lines.append(f"  attempt {i}: {status}, {attempt.latency_ms:.0f}ms")
            for f in attempt.validation_failures:
                lines.append(f"    - [{f.field}] {f.reason}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# ExtractObserver — attach to a forge() call to capture telemetry
# ---------------------------------------------------------------------------

class ExtractObserver:
    """
    Attach to a ``forge()`` call to capture full telemetry.

    Usage::

        observer = ExtractObserver()
        result = forge(prompt=..., schema=..., observer=observer)
        print(observer.report())
    """

    def __init__(self) -> None:
        self._attempts: List[AttemptRecord] = []
        self._start_time: float = time.monotonic()
        self._provider: str = ""
        self._model: str = ""

    def record_attempt(
        self,
        attempt_number: int,
        provider: str,
        model: str,
        raw_response: str,
        parsed_data: Optional[Dict[str, Any]],
        validation_failures: List[ValidationFailure],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        succeeded: bool,
        error: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._attempts.append(
            AttemptRecord(
                attempt_number=attempt_number,
                provider=provider,
                model=model,
                raw_response=raw_response,
                parsed_data=parsed_data,
                validation_failures=validation_failures,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                succeeded=succeeded,
                error=error,
            )
        )

    def report(self) -> ForgeReport:
        winning = next(
            (a.attempt_number for a in self._attempts if a.succeeded), None
        )
        return ForgeReport(
            succeeded=winning is not None,
            total_attempts=len(self._attempts),
            winning_attempt=winning,
            provider=self._provider,
            model=self._model,
            attempts=list(self._attempts),
        )

    def reset(self) -> None:
        """Clear recorded attempts (reuse the same observer for multiple calls)."""
        self._attempts = []
        self._start_time = time.monotonic()
        self._provider = ""
        self._model = ""
