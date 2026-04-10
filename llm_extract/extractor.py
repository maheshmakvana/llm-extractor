"""
extractor.py — Core extraction engine for llm-extract.

Handles:
  - Building prompt messages (system + user)
  - Calling the provider adapter
  - Parsing the raw text response to a dict
  - Running structural + semantic validation
  - Smart correction retries (tells the model exactly what was wrong)
  - Multi-provider fallback chain
  - Returning an ExtractResult with full observability
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, Union

from .core import Schema, SchemaInput, ValidationFailure, build_correction_prompt
from .observability import ExtractObserver
from .providers import ProviderAdapter, ProviderConfig, ProviderResponse, get_adapter, _strip_code_fence


# ---------------------------------------------------------------------------
# ExtractResult
# ---------------------------------------------------------------------------

class ExtractResult:
    """
    Returned by every ``extract()`` / ``aextract()`` call.

    Attributes
    ----------
    data : dict
        The validated, coerced output data.
    succeeded : bool
        False if all retries exhausted without a valid response.
    attempts : int
        How many LLM calls were made.
    provider : str
        Which provider produced the final result.
    model : str
        Which model produced the final result.
    failures : list[ValidationFailure]
        Failures from the last attempt (empty on success).
    raw : str
        Last raw LLM response text.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]],
        succeeded: bool,
        attempts: int,
        provider: str,
        model: str,
        failures: Optional[List[ValidationFailure]] = None,
        raw: str = "",
    ) -> None:
        self.data = data or {}
        self.succeeded = succeeded
        self.attempts = attempts
        self.provider = provider
        self.model = model
        self.failures = failures or []
        self.raw = raw

    def typed_data(self, model_class: Type) -> Any:
        """
        Deserialize ``self.data`` into a pydantic model instance.

        Example::

            result = extract(...)
            product = result.typed_data(Product)
        """
        try:
            return model_class(**self.data)
        except Exception as exc:
            raise ValueError(
                f"Cannot deserialize into {model_class.__name__}: {exc}"
            ) from exc

    def __repr__(self) -> str:
        status = "OK" if self.succeeded else "FAILED"
        return (
            f"ExtractResult({status}, attempts={self.attempts}, "
            f"provider={self.provider!r}, fields={list(self.data.keys())})"
        )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_initial_messages(
    prompt: str,
    schema: Schema,
    system_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_content = (
        "You are a precise data extraction assistant. "
        "Always respond with a valid JSON object and nothing else — "
        "no markdown, no code fences, no commentary.\n\n"
        + schema.to_prompt_description()
    )
    if system_hint:
        system_content += f"\n\n{system_hint}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _build_retry_messages(
    original_messages: List[Dict[str, str]],
    previous_raw: str,
    failures: List[ValidationFailure],
) -> List[Dict[str, str]]:
    correction = build_correction_prompt(failures, previous_raw)
    messages = list(original_messages)
    messages.append({"role": "assistant", "content": previous_raw})
    messages.append({"role": "user", "content": correction})
    return messages


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction from LLM output.
    Handles: clean JSON, code fences, trailing text, leading text.
    """
    if not text or not text.strip():
        return None

    text = text.strip()
    text = _strip_code_fence(text)

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Core extraction logic (sync)
# ---------------------------------------------------------------------------

def _run_extraction(
    adapter: ProviderAdapter,
    messages: List[Dict[str, str]],
    schema: Schema,
    max_retries: int,
    observer: Optional[ExtractObserver],
    provider_name: str,
    model_name: str,
) -> ExtractResult:
    """
    Attempt extraction with up to ``max_retries`` correction passes.
    Returns an ExtractResult (may have succeeded=False if all retries fail).
    """
    json_schema = schema.to_json_schema()
    current_messages = list(messages)
    last_raw = ""
    last_failures: List[ValidationFailure] = []

    for attempt in range(1, max_retries + 1):
        error_str: Optional[str] = None
        provider_resp: Optional[ProviderResponse] = None

        try:
            provider_resp = adapter.complete(current_messages, json_schema=json_schema)
            last_raw = provider_resp.text
        except Exception as exc:
            error_str = str(exc)
            if observer:
                observer.record_attempt(
                    attempt_number=attempt,
                    provider=provider_name,
                    model=model_name,
                    raw_response="",
                    parsed_data=None,
                    validation_failures=[],
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    succeeded=False,
                    error=error_str,
                )
            if attempt == max_retries:
                break
            continue

        parsed = _parse_json(last_raw)

        if parsed is None:
            parse_failure = ValidationFailure("__response__", "Response is not valid JSON", last_raw)
            last_failures = [parse_failure]
            if observer:
                observer.record_attempt(
                    attempt_number=attempt,
                    provider=provider_name,
                    model=model_name,
                    raw_response=last_raw,
                    parsed_data=None,
                    validation_failures=last_failures,
                    input_tokens=provider_resp.input_tokens,
                    output_tokens=provider_resp.output_tokens,
                    latency_ms=provider_resp.latency_ms,
                    succeeded=False,
                )
            current_messages = _build_retry_messages(messages, last_raw, last_failures)
            continue

        coerced = schema.coerce(parsed)
        last_failures = schema.validate_data(coerced)

        if observer:
            observer.record_attempt(
                attempt_number=attempt,
                provider=provider_name,
                model=model_name,
                raw_response=last_raw,
                parsed_data=coerced,
                validation_failures=last_failures,
                input_tokens=provider_resp.input_tokens,
                output_tokens=provider_resp.output_tokens,
                latency_ms=provider_resp.latency_ms,
                succeeded=len(last_failures) == 0,
            )

        if not last_failures:
            return ExtractResult(
                data=coerced,
                succeeded=True,
                attempts=attempt,
                provider=provider_name,
                model=model_name,
                raw=last_raw,
            )

        if attempt < max_retries:
            current_messages = _build_retry_messages(messages, last_raw, last_failures)

    last_parsed = _parse_json(last_raw)
    if last_parsed:
        last_parsed = schema.coerce(last_parsed)

    return ExtractResult(
        data=last_parsed,
        succeeded=False,
        attempts=max_retries,
        provider=provider_name,
        model=model_name,
        failures=last_failures,
        raw=last_raw,
    )


# ---------------------------------------------------------------------------
# Core extraction logic (async)
# ---------------------------------------------------------------------------

async def _run_extraction_async(
    adapter: ProviderAdapter,
    messages: List[Dict[str, str]],
    schema: Schema,
    max_retries: int,
    observer: Optional[ExtractObserver],
    provider_name: str,
    model_name: str,
) -> ExtractResult:
    """Async version of _run_extraction."""
    json_schema = schema.to_json_schema()
    current_messages = list(messages)
    last_raw = ""
    last_failures: List[ValidationFailure] = []

    for attempt in range(1, max_retries + 1):
        error_str: Optional[str] = None
        provider_resp: Optional[ProviderResponse] = None

        try:
            provider_resp = await adapter.acomplete(current_messages, json_schema=json_schema)
            last_raw = provider_resp.text
        except Exception as exc:
            error_str = str(exc)
            if observer:
                observer.record_attempt(
                    attempt_number=attempt,
                    provider=provider_name,
                    model=model_name,
                    raw_response="",
                    parsed_data=None,
                    validation_failures=[],
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    succeeded=False,
                    error=error_str,
                )
            if attempt == max_retries:
                break
            continue

        parsed = _parse_json(last_raw)

        if parsed is None:
            last_failures = [ValidationFailure("__response__", "Response is not valid JSON", last_raw)]
            if observer:
                observer.record_attempt(
                    attempt_number=attempt,
                    provider=provider_name,
                    model=model_name,
                    raw_response=last_raw,
                    parsed_data=None,
                    validation_failures=last_failures,
                    input_tokens=provider_resp.input_tokens,
                    output_tokens=provider_resp.output_tokens,
                    latency_ms=provider_resp.latency_ms,
                    succeeded=False,
                )
            current_messages = _build_retry_messages(messages, last_raw, last_failures)
            continue

        coerced = schema.coerce(parsed)
        last_failures = schema.validate_data(coerced)

        if observer:
            observer.record_attempt(
                attempt_number=attempt,
                provider=provider_name,
                model=model_name,
                raw_response=last_raw,
                parsed_data=coerced,
                validation_failures=last_failures,
                input_tokens=provider_resp.input_tokens,
                output_tokens=provider_resp.output_tokens,
                latency_ms=provider_resp.latency_ms,
                succeeded=len(last_failures) == 0,
            )

        if not last_failures:
            return ExtractResult(
                data=coerced,
                succeeded=True,
                attempts=attempt,
                provider=provider_name,
                model=model_name,
                raw=last_raw,
            )

        if attempt < max_retries:
            current_messages = _build_retry_messages(messages, last_raw, last_failures)

    last_parsed = _parse_json(last_raw)
    if last_parsed:
        last_parsed = schema.coerce(last_parsed)

    return ExtractResult(
        data=last_parsed,
        succeeded=False,
        attempts=max_retries,
        provider=provider_name,
        model=model_name,
        failures=last_failures,
        raw=last_raw,
    )


# ---------------------------------------------------------------------------
# Public extract() and aextract()
# ---------------------------------------------------------------------------

def extract(
    prompt: str,
    schema: SchemaInput,
    provider: str,
    model: str,
    api_key: str,
    *,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 30.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_hint: Optional[str] = None,
    observer: Optional[ExtractObserver] = None,
    fallback_chain: Optional[List[Dict[str, Any]]] = None,
    raise_on_failure: bool = False,
) -> ExtractResult:
    """
    Extract structured output from an LLM with validation and smart retries.

    Parameters
    ----------
    prompt : str
        The user-facing prompt describing what to extract.
    schema : dict | pydantic BaseModel class | JSON Schema dict
        Defines the expected output structure and types.
    provider : str
        ``"openai"``, ``"anthropic"``, ``"gemini"``, or ``"auto"``.
        Use ``"auto"`` together with ``fallback_chain``.
    model : str
        Model ID (e.g. ``"gpt-4o-mini"``, ``"claude-haiku-4-5-20251001"``).
    api_key : str
        Provider API key.
    base_url : str, optional
        Override for OpenAI-compatible endpoints.
    max_retries : int
        Maximum LLM calls per provider (default 3).
    timeout : float
        HTTP timeout in seconds (default 30).
    temperature : float
        Sampling temperature (default 0 for determinism).
    max_output_tokens : int
        Maximum tokens in the response (default 1024).
    system_hint : str, optional
        Extra text appended to the system prompt.
    observer : ExtractObserver, optional
        Attach to capture full telemetry.
    fallback_chain : list[dict], optional
        List of ``{"provider", "model", "api_key"}`` dicts tried in order
        when ``provider="auto"`` or after all retries on the primary fail.
    raise_on_failure : bool
        If True, raise ``ExtractValidationError`` when all retries fail.

    Returns
    -------
    ExtractResult
    """
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    messages = _build_initial_messages(prompt, schema, system_hint)

    if provider.lower() != "auto" and not fallback_chain:
        config = ProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        adapter = get_adapter(config)
        result = _run_extraction(adapter, messages, schema, max_retries, observer, provider, model)

        if not result.succeeded and raise_on_failure:
            raise ExtractValidationError(result)
        return result

    chain = fallback_chain or []
    if provider.lower() != "auto":
        primary: Dict[str, Any] = {"provider": provider, "model": model, "api_key": api_key}
        if base_url:
            primary["base_url"] = base_url
        chain = [primary] + list(chain)

    result = None
    for entry in chain:
        p = entry["provider"]
        m = entry["model"]
        k = entry["api_key"]
        bu = entry.get("base_url")

        config = ProviderConfig(
            provider=p,
            model=m,
            api_key=k,
            base_url=bu,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        adapter = get_adapter(config)
        result = _run_extraction(adapter, messages, schema, max_retries, observer, p, m)
        if result.succeeded:
            return result

    if raise_on_failure and result is not None:
        raise ExtractValidationError(result)
    return result  # type: ignore[return-value]


async def aextract(
    prompt: str,
    schema: SchemaInput,
    provider: str,
    model: str,
    api_key: str,
    *,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 30.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_hint: Optional[str] = None,
    observer: Optional[ExtractObserver] = None,
    fallback_chain: Optional[List[Dict[str, Any]]] = None,
    raise_on_failure: bool = False,
) -> ExtractResult:
    """Async version of ``extract()``. See ``extract()`` for full parameter docs."""
    if not isinstance(schema, Schema):
        schema = Schema(schema)

    messages = _build_initial_messages(prompt, schema, system_hint)

    if provider.lower() != "auto" and not fallback_chain:
        config = ProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        adapter = get_adapter(config)
        result = await _run_extraction_async(adapter, messages, schema, max_retries, observer, provider, model)

        if not result.succeeded and raise_on_failure:
            raise ExtractValidationError(result)
        return result

    chain = fallback_chain or []
    if provider.lower() != "auto":
        primary: Dict[str, Any] = {"provider": provider, "model": model, "api_key": api_key}
        if base_url:
            primary["base_url"] = base_url
        chain = [primary] + list(chain)

    result = None
    for entry in chain:
        p = entry["provider"]
        m = entry["model"]
        k = entry["api_key"]
        bu = entry.get("base_url")

        config = ProviderConfig(
            provider=p,
            model=m,
            api_key=k,
            base_url=bu,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        adapter = get_adapter(config)
        result = await _run_extraction_async(adapter, messages, schema, max_retries, observer, p, m)
        if result.succeeded:
            return result

    if raise_on_failure and result is not None:
        raise ExtractValidationError(result)
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExtractValidationError(Exception):
    """Raised by extract() when raise_on_failure=True and all retries fail."""

    def __init__(self, result: ExtractResult) -> None:
        self.result = result
        failures_str = "; ".join(f"{f.field}: {f.reason}" for f in result.failures)
        super().__init__(
            f"All {result.attempts} attempts failed validation. "
            f"Last failures: {failures_str or 'no parseable data'}"
        )
