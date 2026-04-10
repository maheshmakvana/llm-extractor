"""
llm_extract.advanced — Advanced extraction utilities.

New in 1.1.0:
- ExtractionCache: Hash-based response cache to avoid redundant LLM calls
- batch_extract / abatch_extract: Extract from many prompts concurrently
- ConfidenceScorer: Estimate confidence of an extraction result
- SchemaEvolver: Migrate extracted data between schema versions
- ExtractionPipeline: Multi-stage extraction with chained schemas
- RateLimiter: Token-bucket rate limiter for API compliance
- extract_with_retry_budget: Budget-aware extraction with timeout

New in 1.2.0:
- OutputTransformer: Post-process / normalise extraction results with chained transforms
- FieldConfidenceScorer: Per-field confidence scores (not just overall)
- PartialExtractor: Return best-effort partial data even when validation fails
- ExtractionDiff: Diff two ExtractResult objects to detect field-level changes
- MultiSchemaExtractor: Run one prompt against N schemas concurrently and merge
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import threading
from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .core import Schema, SchemaInput, ValidationFailure
from .observability import ExtractObserver


# ---------------------------------------------------------------------------
# ExtractionCache
# ---------------------------------------------------------------------------

class ExtractionCache:
    """
    Thread-safe, LRU, hash-keyed cache for extraction results.

    Avoids re-calling the LLM for identical (prompt, schema, provider, model) combos.

    Parameters
    ----------
    maxsize : int
        Maximum cached entries.
    ttl : float | None
        Seconds before an entry expires; None = no expiry.

    Example
    -------
    >>> cache = ExtractionCache(maxsize=256, ttl=3600)
    >>> result = extract(..., cache=cache)    # cache-miss: calls LLM
    >>> result = extract(..., cache=cache)    # cache-hit: instant
    """

    def __init__(self, maxsize: int = 256, ttl: Optional[float] = None) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(prompt: str, schema_json: Dict, provider: str, model: str) -> str:
        raw = json.dumps(
            {"p": prompt, "s": schema_json, "pr": provider, "m": model},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            entry = self._store[key]
            if self._ttl and (time.time() - entry["ts"]) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            entry["hits"] += 1
            self._hits += 1
            return entry["data"]

    def set(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._store) >= self._maxsize and key not in self._store:
                self._store.popitem(last=False)
            self._store[key] = {"data": data, "ts": time.time(), "hits": 0}
            if key in self._store:
                self._store.move_to_end(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        with self._lock:
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


# ---------------------------------------------------------------------------
# RateLimiter  (token-bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter for LLM API compliance.

    Parameters
    ----------
    calls_per_minute : int
        Maximum calls allowed per 60-second window.

    Example
    -------
    >>> limiter = RateLimiter(calls_per_minute=60)
    >>> for prompt in prompts:
    ...     limiter.acquire()          # blocks if over limit
    ...     extract(prompt, ...)
    """

    def __init__(self, calls_per_minute: int) -> None:
        self._cpm = calls_per_minute
        self._interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """Block until a slot is available."""
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    async def async_acquire(self) -> None:
        """Async version of acquire()."""
        now = time.monotonic()
        wait = self._interval - (now - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# batch_extract
# ---------------------------------------------------------------------------

def batch_extract(
    prompts: List[str],
    schema: SchemaInput,
    provider: str,
    model: str,
    api_key: str,
    *,
    max_workers: int = 4,
    max_retries: int = 3,
    timeout: float = 30.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_hint: Optional[str] = None,
    observer: Optional[ExtractObserver] = None,
    cache: Optional[ExtractionCache] = None,
    rate_limiter: Optional[RateLimiter] = None,
    on_result: Optional[Callable] = None,
) -> List[Any]:
    """
    Extract structured data from multiple prompts concurrently.

    Parameters
    ----------
    prompts : list[str]
        Prompts to process.
    max_workers : int
        Thread pool size (default 4).
    on_result : callable | None
        Called with (index, result) as each extraction completes.

    Returns
    -------
    List of ExtractResult objects in the same order as *prompts*.
    """
    import concurrent.futures
    from .extractor import extract as _extract

    results: List[Any] = [None] * len(prompts)

    def _run(idx: int, prompt: str):
        if rate_limiter:
            rate_limiter.acquire()

        # cache check
        if cache is not None:
            s_obj = Schema(schema) if not isinstance(schema, Schema) else schema
            key = ExtractionCache.make_key(prompt, s_obj.to_json_schema(), provider, model)
            cached = cache.get(key)
            if cached is not None:
                from .extractor import ExtractResult
                result = ExtractResult(
                    data=cached, succeeded=True, attempts=0,
                    provider=provider, model=model, raw="[cache-hit]"
                )
                if on_result:
                    on_result(idx, result)
                return result

        result = _extract(
            prompt=prompt,
            schema=schema,
            provider=provider,
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_hint=system_hint,
            observer=observer,
        )

        if cache is not None and result.succeeded:
            s_obj = Schema(schema) if not isinstance(schema, Schema) else schema
            key = ExtractionCache.make_key(prompt, s_obj.to_json_schema(), provider, model)
            cache.set(key, result.data)

        if on_result:
            on_result(idx, result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, i, p): i for i, p in enumerate(prompts)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


async def abatch_extract(
    prompts: List[str],
    schema: SchemaInput,
    provider: str,
    model: str,
    api_key: str,
    *,
    max_concurrent: int = 4,
    max_retries: int = 3,
    timeout: float = 30.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_hint: Optional[str] = None,
    cache: Optional[ExtractionCache] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> List[Any]:
    """
    Async version of batch_extract using asyncio semaphore for concurrency control.
    """
    from .extractor import aextract as _aextract, ExtractResult

    sem = asyncio.Semaphore(max_concurrent)

    async def _run(idx: int, prompt: str):
        async with sem:
            if rate_limiter:
                await rate_limiter.async_acquire()

            if cache is not None:
                s_obj = Schema(schema) if not isinstance(schema, Schema) else schema
                key = ExtractionCache.make_key(prompt, s_obj.to_json_schema(), provider, model)
                cached = cache.get(key)
                if cached is not None:
                    return idx, ExtractResult(
                        data=cached, succeeded=True, attempts=0,
                        provider=provider, model=model, raw="[cache-hit]"
                    )

            result = await _aextract(
                prompt=prompt,
                schema=schema,
                provider=provider,
                model=model,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_hint=system_hint,
            )

            if cache is not None and result.succeeded:
                s_obj = Schema(schema) if not isinstance(schema, Schema) else schema
                key = ExtractionCache.make_key(prompt, s_obj.to_json_schema(), provider, model)
                cache.set(key, result.data)

            return idx, result

    tasks = [_run(i, p) for i, p in enumerate(prompts)]
    pairs = await asyncio.gather(*tasks)
    results: List[Any] = [None] * len(prompts)
    for idx, result in pairs:
        results[idx] = result
    return results


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Estimate a confidence score [0.0, 1.0] for an extraction result.

    The score reflects structural completeness, field type correctness,
    and retry count (fewer retries = higher confidence).

    Example
    -------
    >>> scorer = ConfidenceScorer()
    >>> score = scorer.score(result, schema)
    >>> print(f"Confidence: {score:.2%}")
    """

    def score(self, result: Any, schema: SchemaInput) -> float:
        """
        Return a confidence score between 0.0 and 1.0.

        Parameters
        ----------
        result : ExtractResult
        schema : SchemaInput
        """
        if not result.succeeded:
            return 0.0

        s = Schema(schema) if not isinstance(schema, Schema) else schema
        fields = s.fields
        total = len(fields)
        if total == 0:
            return 1.0 if result.succeeded else 0.0

        data = result.data or {}
        present = sum(1 for f in fields if f.name in data and data[f.name] is not None)
        type_ok = 0
        for f in fields:
            val = data.get(f.name)
            if val is None:
                continue
            try:
                if isinstance(val, f.python_type):
                    type_ok += 1
                elif f.python_type == float and isinstance(val, (int, float)):
                    type_ok += 1
            except Exception:
                pass

        completeness = present / total
        type_score = type_ok / total
        retry_penalty = max(0.0, 1.0 - (result.attempts - 1) * 0.15)
        raw_score = (0.5 * completeness + 0.35 * type_score + 0.15 * retry_penalty)
        return round(min(1.0, max(0.0, raw_score)), 4)

    def label(self, score: float) -> str:
        if score >= 0.9:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# SchemaEvolver
# ---------------------------------------------------------------------------

class SchemaEvolver:
    """
    Migrate extracted data between schema versions using registered transforms.

    Use when your schema changes over time and you need to handle data
    extracted under old schemas.

    Example
    -------
    >>> evolver = SchemaEvolver()
    >>> evolver.register("v1", "v2", lambda d: {**d, "full_name": d.pop("name", "")})
    >>> v2_data = evolver.migrate(v1_data, from_version="v1", to_version="v2")
    """

    def __init__(self) -> None:
        self._transforms: Dict[Tuple[str, str], Callable] = {}

    def register(
        self,
        from_version: str,
        to_version: str,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> "SchemaEvolver":
        """Register a migration function from one schema version to another."""
        self._transforms[(from_version, to_version)] = transform
        return self

    def migrate(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """
        Apply a registered migration transform.

        Raises
        ------
        KeyError
            If no transform is registered for the given version pair.
        """
        key = (from_version, to_version)
        if key not in self._transforms:
            raise KeyError(
                f"No migration registered from '{from_version}' → '{to_version}'. "
                f"Available: {list(self._transforms.keys())}"
            )
        return self._transforms[key](dict(data))

    def list_migrations(self) -> List[Tuple[str, str]]:
        return list(self._transforms.keys())


# ---------------------------------------------------------------------------
# ExtractionPipeline
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """
    Multi-stage extraction pipeline: chain multiple schemas sequentially.

    Each stage extracts from the *previous stage's result* formatted as text.
    Useful for complex documents requiring hierarchical extraction.

    Example
    -------
    >>> pipe = ExtractionPipeline(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    >>> pipe.add_stage("entities", entity_schema)
    >>> pipe.add_stage("sentiment", sentiment_schema)
    >>> results = pipe.run("Full document text here...")
    >>> results["entities"]  # ExtractResult from stage 1
    >>> results["sentiment"] # ExtractResult from stage 2
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout
        self._temperature = temperature
        self._stages: List[Tuple[str, SchemaInput, Optional[str]]] = []

    def add_stage(
        self,
        name: str,
        schema: SchemaInput,
        prompt_template: Optional[str] = None,
    ) -> "ExtractionPipeline":
        """
        Add a stage.

        Parameters
        ----------
        name : str
            Stage identifier.
        schema : SchemaInput
            Schema for this stage.
        prompt_template : str | None
            Template with {input} placeholder. Defaults to the raw input text.
        """
        self._stages.append((name, schema, prompt_template))
        return self

    def run(self, text: str) -> Dict[str, Any]:
        """
        Run all stages on *text*. Returns dict mapping stage name → ExtractResult.
        """
        from .extractor import extract as _extract

        results: Dict[str, Any] = {}
        current_text = text

        for name, schema, template in self._stages:
            prompt = template.format(input=current_text) if template else current_text
            result = _extract(
                prompt=prompt,
                schema=schema,
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
                timeout=self._timeout,
                temperature=self._temperature,
            )
            results[name] = result
            # Feed this stage's extracted data as context for the next stage
            if result.succeeded and result.data:
                current_text = f"Previous extraction: {json.dumps(result.data)}\n\nOriginal text: {text}"

        return results

    async def arun(self, text: str) -> Dict[str, Any]:
        """Async version of run()."""
        from .extractor import aextract as _aextract

        results: Dict[str, Any] = {}
        current_text = text

        for name, schema, template in self._stages:
            prompt = template.format(input=current_text) if template else current_text
            result = await _aextract(
                prompt=prompt,
                schema=schema,
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
                timeout=self._timeout,
                temperature=self._temperature,
            )
            results[name] = result
            if result.succeeded and result.data:
                current_text = f"Previous extraction: {json.dumps(result.data)}\n\nOriginal text: {text}"

        return results


# ---------------------------------------------------------------------------
# extract_with_budget (timeout-aware)
# ---------------------------------------------------------------------------

def extract_with_budget(
    prompt: str,
    schema: SchemaInput,
    provider: str,
    model: str,
    api_key: str,
    *,
    total_budget_seconds: float = 10.0,
    max_retries: int = 3,
    timeout_per_attempt: float = 8.0,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_hint: Optional[str] = None,
) -> Any:
    """
    Extract with a hard wall-clock budget. Returns a partial or failed result
    if the budget is exhausted before success.

    Parameters
    ----------
    total_budget_seconds : float
        Maximum total wall-clock seconds for all attempts.
    timeout_per_attempt : float
        Per-attempt HTTP timeout (should be < total_budget_seconds).
    """
    from .extractor import extract as _extract

    deadline = time.monotonic() + total_budget_seconds
    attempt = 0
    last_result = None

    while time.monotonic() < deadline and attempt < max_retries:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        per_timeout = min(timeout_per_attempt, remaining)
        result = _extract(
            prompt=prompt,
            schema=schema,
            provider=provider,
            model=model,
            api_key=api_key,
            max_retries=1,
            timeout=per_timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_hint=system_hint,
        )
        last_result = result
        attempt += 1
        if result.succeeded:
            return result

    return last_result


# ---------------------------------------------------------------------------
# OutputTransformer  (new in 1.2.0)
# ---------------------------------------------------------------------------

class OutputTransformer:
    """
    Chain post-processing transforms over an ExtractResult's data dict.

    Each transform is a callable ``(data: dict) -> dict``.
    Transforms run in the order they are registered.

    Example
    -------
    >>> transformer = OutputTransformer()
    >>> transformer.add(lambda d: {**d, "name": d.get("name", "").strip().title()})
    >>> transformer.add(lambda d: {**d, "age": max(0, d.get("age", 0))})
    >>> result = extract(...)
    >>> clean = transformer.apply(result)
    >>> clean.data  # transformed dict
    """

    def __init__(self) -> None:
        self._transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

    def add(self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "OutputTransformer":
        """Register a transform. Returns self for chaining."""
        self._transforms.append(transform)
        return self

    def apply(self, result: Any) -> Any:
        """
        Apply all transforms to result.data. Returns a new ExtractResult with
        the transformed data (does not mutate the original).
        """
        import copy
        from .extractor import ExtractResult

        new_data = copy.deepcopy(result.data or {})
        for fn in self._transforms:
            try:
                new_data = fn(new_data)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("OutputTransformer: transform %s raised %s", fn, exc)

        return ExtractResult(
            data=new_data,
            succeeded=result.succeeded,
            attempts=result.attempts,
            provider=result.provider,
            model=result.model,
            failures=result.failures,
            raw=result.raw,
        )

    def apply_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to a plain dict (no ExtractResult needed)."""
        import copy
        result = copy.deepcopy(data)
        for fn in self._transforms:
            result = fn(result)
        return result


# ---------------------------------------------------------------------------
# FieldConfidenceScorer  (new in 1.2.0)
# ---------------------------------------------------------------------------

class FieldConfidenceScorer:
    """
    Compute per-field confidence scores for an extraction result.

    Returns a dict mapping field names to scores in [0.0, 1.0]:
      - 1.0  field present with correct type
      - 0.5  field present but type is wrong / unexpected
      - 0.0  field missing or None

    Example
    -------
    >>> scorer = FieldConfidenceScorer()
    >>> scores = scorer.score(result, schema)
    >>> print(scores)
    {'name': 1.0, 'age': 1.0, 'email': 0.5}
    >>> print(scorer.overall(scores))
    0.833
    """

    def score(self, result: Any, schema: "SchemaInput") -> Dict[str, float]:
        """Return per-field confidence scores."""
        s = Schema(schema) if not isinstance(schema, Schema) else schema
        data = result.data or {}
        scores: Dict[str, float] = {}

        for field in s.fields:
            val = data.get(field.name)
            if val is None:
                scores[field.name] = 0.0
            else:
                try:
                    if isinstance(val, field.python_type):
                        scores[field.name] = 1.0
                    elif field.python_type == float and isinstance(val, (int, float)):
                        scores[field.name] = 1.0
                    else:
                        scores[field.name] = 0.5
                except Exception:
                    scores[field.name] = 0.5

        return scores

    def overall(self, scores: Dict[str, float]) -> float:
        """Average of all per-field scores. Returns 0.0 for empty dicts."""
        if not scores:
            return 0.0
        return round(sum(scores.values()) / len(scores), 4)

    def low_confidence_fields(self, scores: Dict[str, float], threshold: float = 0.5) -> List[str]:
        """Return field names whose confidence is strictly below threshold."""
        return [k for k, v in scores.items() if v < threshold]


# ---------------------------------------------------------------------------
# PartialExtractor  (new in 1.2.0)
# ---------------------------------------------------------------------------

class PartialExtractor:
    """
    Return the best-effort partial data even when full validation fails.

    Standard extract() returns succeeded=False and partial data when
    validation fails. PartialExtractor makes this first-class: it always
    returns whatever fields were successfully extracted, annotating each
    field with a ``_partial`` flag in the result metadata.

    Example
    -------
    >>> partial = PartialExtractor(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    >>> result = partial.extract("John is 34 years old.", {"name": str, "age": int, "email": str})
    >>> print(result.data)        # e.g. {'name': 'John', 'age': 34}  — email missing but still returned
    >>> print(result.partial_fields)  # ['email']
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout
        self._temperature = temperature

    def extract(self, prompt: str, schema: "SchemaInput") -> "PartialResult":
        from .extractor import extract as _extract
        result = _extract(
            prompt=prompt,
            schema=schema,
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=self._max_retries,
            timeout=self._timeout,
            temperature=self._temperature,
        )
        failed_fields = {f.field for f in result.failures}
        data = {k: v for k, v in (result.data or {}).items() if k not in failed_fields}
        return PartialResult(data=data, full_result=result, partial_fields=list(failed_fields))

    async def aextract(self, prompt: str, schema: "SchemaInput") -> "PartialResult":
        from .extractor import aextract as _aextract
        result = await _aextract(
            prompt=prompt,
            schema=schema,
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=self._max_retries,
            timeout=self._timeout,
            temperature=self._temperature,
        )
        failed_fields = {f.field for f in result.failures}
        data = {k: v for k, v in (result.data or {}).items() if k not in failed_fields}
        return PartialResult(data=data, full_result=result, partial_fields=list(failed_fields))


class PartialResult:
    """Result from PartialExtractor — always contains the best-effort data."""

    __slots__ = ("data", "full_result", "partial_fields")

    def __init__(self, data: Dict[str, Any], full_result: Any, partial_fields: List[str]) -> None:
        self.data = data
        self.full_result = full_result
        self.partial_fields = partial_fields

    @property
    def is_complete(self) -> bool:
        return len(self.partial_fields) == 0

    def __repr__(self) -> str:
        return (
            f"PartialResult(fields={list(self.data.keys())}, "
            f"partial={self.partial_fields}, complete={self.is_complete})"
        )


# ---------------------------------------------------------------------------
# ExtractionDiff  (new in 1.2.0)
# ---------------------------------------------------------------------------

class ExtractionDiff:
    """
    Diff two ExtractResult objects to identify field-level changes.

    Useful for regression testing, schema migration checks, or monitoring
    prompt changes that alter extraction output.

    Example
    -------
    >>> diff = ExtractionDiff(old_result, new_result)
    >>> print(diff.changed)   # {'age': (34, 35)}
    >>> print(diff.added)     # {'email': 'foo@bar.com'}
    >>> print(diff.removed)   # {'nickname': 'JD'}
    >>> print(diff.summary()) # human-readable summary
    """

    def __init__(self, old: Any, new: Any) -> None:
        old_data = old.data or {}
        new_data = new.data or {}

        old_keys = set(old_data.keys())
        new_keys = set(new_data.keys())

        self.added: Dict[str, Any] = {k: new_data[k] for k in new_keys - old_keys}
        self.removed: Dict[str, Any] = {k: old_data[k] for k in old_keys - new_keys}
        self.changed: Dict[str, Tuple[Any, Any]] = {
            k: (old_data[k], new_data[k])
            for k in old_keys & new_keys
            if old_data[k] != new_data[k]
        }
        self.unchanged: Dict[str, Any] = {
            k: old_data[k]
            for k in old_keys & new_keys
            if old_data[k] == new_data[k]
        }

    @property
    def has_diff(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> str:
        lines = [f"ExtractionDiff — {'changes detected' if self.has_diff else 'no changes'}"]
        if self.added:
            lines.append(f"  Added:   {list(self.added.keys())}")
        if self.removed:
            lines.append(f"  Removed: {list(self.removed.keys())}")
        if self.changed:
            for k, (old_v, new_v) in self.changed.items():
                lines.append(f"  Changed: {k!r}: {old_v!r} → {new_v!r}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": {k: {"old": o, "new": n} for k, (o, n) in self.changed.items()},
            "unchanged_count": len(self.unchanged),
        }


# ---------------------------------------------------------------------------
# MultiSchemaExtractor  (new in 1.2.0)
# ---------------------------------------------------------------------------

class MultiSchemaExtractor:
    """
    Run one prompt against multiple schemas concurrently and collect results.

    Useful when the same document needs to be parsed with different schemas
    simultaneously (e.g. extracting entities AND sentiment AND metadata).

    Parameters
    ----------
    provider : str
    model : str
    api_key : str
    max_workers : int
        Thread pool size for concurrent schema extraction.

    Example
    -------
    >>> mse = MultiSchemaExtractor("openai", "gpt-4o-mini", api_key="sk-...")
    >>> results = mse.run("John Doe, 34, likes hiking.", {
    ...     "person": {"name": str, "age": int},
    ...     "interests": {"hobby": str},
    ... })
    >>> results["person"].data    # {'name': 'John Doe', 'age': 34}
    >>> results["interests"].data # {'hobby': 'hiking'}
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        max_workers: int = 4,
        max_retries: int = 3,
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._timeout = timeout
        self._temperature = temperature

    def run(
        self,
        prompt: str,
        schemas: Dict[str, "SchemaInput"],
    ) -> Dict[str, Any]:
        """
        Extract prompt against each schema concurrently.

        Parameters
        ----------
        prompt : str
        schemas : dict[str, SchemaInput]
            Mapping of label → schema.

        Returns
        -------
        dict[str, ExtractResult]
        """
        import concurrent.futures
        from .extractor import extract as _extract

        results: Dict[str, Any] = {}

        def _run_one(label: str, schema: "SchemaInput") -> tuple:
            result = _extract(
                prompt=prompt,
                schema=schema,
                provider=self._provider,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
                timeout=self._timeout,
                temperature=self._temperature,
            )
            return label, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futs = {ex.submit(_run_one, label, schema): label for label, schema in schemas.items()}
            for fut in concurrent.futures.as_completed(futs):
                label, result = fut.result()
                results[label] = result

        return results

    async def arun(
        self,
        prompt: str,
        schemas: Dict[str, "SchemaInput"],
    ) -> Dict[str, Any]:
        """Async version of run()."""
        from .extractor import aextract as _aextract

        sem = asyncio.Semaphore(self._max_workers)

        async def _run_one(label: str, schema: "SchemaInput") -> tuple:
            async with sem:
                result = await _aextract(
                    prompt=prompt,
                    schema=schema,
                    provider=self._provider,
                    model=self._model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    max_retries=self._max_retries,
                    timeout=self._timeout,
                    temperature=self._temperature,
                )
                return label, result

        pairs = await asyncio.gather(*[_run_one(l, s) for l, s in schemas.items()])
        return dict(pairs)
