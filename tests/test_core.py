"""
tests/test_core.py — Unit tests for llm-extract (no API keys required).

Covers:
  - Schema construction (dict, pydantic, JSON Schema)
  - SemanticRule validation (min/max, allowed_values, pattern, custom)
  - Type coercion
  - ValidationFailure reporting
  - build_correction_prompt
  - JSON parsing helpers (_parse_json, _strip_code_fence)
  - ExtractResult / ExtractObserver
  - extract() with a mock adapter (no real LLM calls)
"""
import json
import pytest

from llm_extract.core import (
    Schema,
    SemanticRule,
    ValidationFailure,
    build_correction_prompt,
    _coerce_value,
)
from llm_extract.extractor import (
    ExtractResult,
    ExtractValidationError,
    _parse_json,
    _build_initial_messages,
    _build_retry_messages,
    _run_extraction,
)
from llm_extract.observability import ExtractObserver
from llm_extract.providers import ProviderAdapter, ProviderResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAdapter(ProviderAdapter):
    """Returns a preset sequence of responses for testing."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def complete(self, messages, json_schema=None):
        text = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return ProviderResponse(
            text=text,
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
            provider="mock",
            model="mock-model",
        )


# ---------------------------------------------------------------------------
# Schema — dict-based
# ---------------------------------------------------------------------------

class TestSchemaDictBased:
    def test_basic_fields_parsed(self):
        schema = Schema({"name": str, "age": int, "score": float})
        names = schema.field_names
        assert "name" in names
        assert "age" in names
        assert "score" in names

    def test_to_json_schema_structure(self):
        schema = Schema({"title": str, "year": int})
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "title" in js["properties"]
        assert "year" in js["properties"]
        assert set(js["required"]) == {"title", "year"}

    def test_invalid_field_type_raises(self):
        with pytest.raises(ValueError, match="Python type"):
            Schema({"name": "not_a_type"})

    def test_validate_valid_data(self):
        schema = Schema({"name": str, "count": int})
        failures = schema.validate_data({"name": "Alice", "count": 5})
        assert failures == []

    def test_validate_missing_required_field(self):
        schema = Schema({"name": str, "count": int})
        failures = schema.validate_data({"name": "Alice"})
        assert any(f.field == "count" for f in failures)

    def test_validate_wrong_type(self):
        schema = Schema({"count": int})
        failures = schema.validate_data({"count": "not-a-number"})
        assert any(f.field == "count" for f in failures)

    def test_float_field_accepts_int_value(self):
        schema = Schema({"ratio": float})
        failures = schema.validate_data({"ratio": 1})
        assert failures == []

    def test_bool_does_not_satisfy_int_field(self):
        schema = Schema({"count": int})
        failures = schema.validate_data({"count": True})
        assert any(f.field == "count" for f in failures)

    def test_prompt_description_contains_schema(self):
        schema = Schema({"x": int})
        desc = schema.to_prompt_description()
        assert "JSON" in desc
        assert "x" in desc


# ---------------------------------------------------------------------------
# Schema — raw JSON Schema
# ---------------------------------------------------------------------------

class TestSchemaJsonSchema:
    def test_json_schema_parsed(self):
        js = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["title", "year"],
        }
        schema = Schema(js)
        assert "title" in schema.field_names
        assert "year" in schema.field_names

    def test_json_schema_roundtrip(self):
        js = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        }
        schema = Schema(js)
        out = schema.to_json_schema()
        assert out["type"] == "object"


# ---------------------------------------------------------------------------
# Schema — pydantic (optional)
# ---------------------------------------------------------------------------

class TestSchemaPydantic:
    def test_pydantic_model(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        schema = Schema(Person)
        assert "name" in schema.field_names
        assert "age" in schema.field_names

    def test_pydantic_validation_failure(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        schema = Schema(Person)
        failures = schema.validate_data({"name": "Alice", "age": "not-int"})
        assert len(failures) > 0


# ---------------------------------------------------------------------------
# SemanticRule
# ---------------------------------------------------------------------------

class TestSemanticRule:
    def test_min_value_pass(self):
        rule = SemanticRule("age", min_value=0)
        ok, _ = rule.check(5)
        assert ok

    def test_min_value_fail(self):
        rule = SemanticRule("age", min_value=0)
        ok, reason = rule.check(-1)
        assert not ok
        assert "minimum" in reason or "below" in reason

    def test_max_value_pass(self):
        rule = SemanticRule("score", max_value=100)
        ok, _ = rule.check(99.9)
        assert ok

    def test_max_value_fail(self):
        rule = SemanticRule("score", max_value=100)
        ok, reason = rule.check(101)
        assert not ok
        assert "maximum" in reason or "exceeds" in reason

    def test_allowed_values_pass(self):
        rule = SemanticRule("status", allowed_values=["active", "inactive"])
        ok, _ = rule.check("active")
        assert ok

    def test_allowed_values_fail(self):
        rule = SemanticRule("status", allowed_values=["active", "inactive"])
        ok, reason = rule.check("deleted")
        assert not ok
        assert "allowed" in reason

    def test_pattern_pass(self):
        rule = SemanticRule("email", pattern=r"^[^@]+@[^@]+\.[^@]+$")
        ok, _ = rule.check("user@example.com")
        assert ok

    def test_pattern_fail(self):
        rule = SemanticRule("email", pattern=r"^[^@]+@[^@]+\.[^@]+$")
        ok, reason = rule.check("not-an-email")
        assert not ok

    def test_custom_validator_pass(self):
        rule = SemanticRule("n", validator=lambda v: v % 2 == 0, message="must be even")
        ok, _ = rule.check(4)
        assert ok

    def test_custom_validator_fail(self):
        rule = SemanticRule("n", validator=lambda v: v % 2 == 0, message="must be even")
        ok, reason = rule.check(3)
        assert not ok
        assert "even" in reason

    def test_custom_message_used(self):
        rule = SemanticRule("x", min_value=10, message="x must be big enough")
        ok, reason = rule.check(1)
        assert not ok
        assert "big enough" in reason

    def test_none_value_passes(self):
        rule = SemanticRule("x", min_value=0)
        ok, _ = rule.check(None)
        assert ok

    def test_schema_add_rule_chaining(self):
        schema = (
            Schema({"age": int, "score": float})
            .add_rule(SemanticRule("age", min_value=0, max_value=150))
            .add_rule(SemanticRule("score", min_value=0.0, max_value=100.0))
        )
        failures = schema.validate_data({"age": 200, "score": 50.0})
        assert any(f.field == "age" for f in failures)
        assert not any(f.field == "score" for f in failures)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

class TestCoercion:
    def test_string_to_int(self):
        assert _coerce_value("42", int) == 42

    def test_string_to_float(self):
        assert _coerce_value("3.14", float) == pytest.approx(3.14)

    def test_int_to_float(self):
        assert _coerce_value(1, float) == 1.0

    def test_bool_string_true(self):
        assert _coerce_value("true", bool) is True

    def test_bool_string_false(self):
        assert _coerce_value("false", bool) is False

    def test_invalid_coercion_returns_original(self):
        result = _coerce_value("abc", int)
        assert result == "abc"

    def test_schema_coerce_applies_to_fields(self):
        schema = Schema({"count": int, "ratio": float})
        coerced = schema.coerce({"count": "5", "ratio": "0.5"})
        assert coerced["count"] == 5
        assert coerced["ratio"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_correction_prompt
# ---------------------------------------------------------------------------

class TestCorrectionPrompt:
    def test_contains_failures(self):
        failures = [
            ValidationFailure("age", "below minimum 0", -5),
            ValidationFailure("email", "does not match pattern", "bad"),
        ]
        prompt = build_correction_prompt(failures, '{"age": -5, "email": "bad"}')
        assert "age" in prompt
        assert "email" in prompt
        assert "minimum" in prompt
        assert "pattern" in prompt

    def test_contains_raw_response(self):
        failures = [ValidationFailure("x", "missing")]
        raw = '{"wrong": 1}'
        prompt = build_correction_prompt(failures, raw)
        assert raw in prompt

    def test_no_markdown_instruction(self):
        failures = [ValidationFailure("x", "bad")]
        prompt = build_correction_prompt(failures, "{}")
        assert "no extra text" in prompt or "ONLY" in prompt


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_clean_json(self):
        result = _parse_json('{"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}

    def test_json_with_code_fence(self):
        result = _parse_json('```json\n{"x": 1}\n```')
        assert result == {"x": 1}

    def test_json_with_plain_fence(self):
        result = _parse_json('```\n{"x": 2}\n```')
        assert result == {"x": 2}

    def test_json_embedded_in_text(self):
        result = _parse_json('Here is the data: {"key": "value"} done.')
        assert result == {"key": "value"}

    def test_empty_string_returns_none(self):
        assert _parse_json("") is None

    def test_plain_text_returns_none(self):
        assert _parse_json("This is not JSON at all.") is None

    def test_array_returns_none(self):
        # We only accept objects
        assert _parse_json("[1, 2, 3]") is None

    def test_nested_json(self):
        result = _parse_json('{"outer": {"inner": 42}}')
        assert result["outer"]["inner"] == 42


# ---------------------------------------------------------------------------
# ExtractResult
# ---------------------------------------------------------------------------

class TestExtractResult:
    def test_repr_ok(self):
        r = ExtractResult({"name": "A"}, succeeded=True, attempts=1, provider="openai", model="m")
        assert "OK" in repr(r)

    def test_repr_failed(self):
        r = ExtractResult(None, succeeded=False, attempts=3, provider="openai", model="m")
        assert "FAILED" in repr(r)

    def test_typed_data(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        r = ExtractResult({"name": "Bob", "age": 25}, succeeded=True, attempts=1, provider="p", model="m")
        person = r.typed_data(Person)
        assert person.name == "Bob"
        assert person.age == 25

    def test_typed_data_raises_on_bad_data(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Strict(BaseModel):
            value: int

        r = ExtractResult({"value": "not-int"}, succeeded=True, attempts=1, provider="p", model="m")
        with pytest.raises(ValueError):
            r.typed_data(Strict)


# ---------------------------------------------------------------------------
# ExtractObserver
# ---------------------------------------------------------------------------

class TestExtractObserver:
    def test_report_empty(self):
        obs = ExtractObserver()
        report = obs.report()
        assert report.total_attempts == 0
        assert not report.succeeded

    def test_record_success(self):
        obs = ExtractObserver()
        obs.record_attempt(
            attempt_number=1,
            provider="openai",
            model="gpt-4o-mini",
            raw_response='{"x": 1}',
            parsed_data={"x": 1},
            validation_failures=[],
            input_tokens=10,
            output_tokens=5,
            latency_ms=200.0,
            succeeded=True,
        )
        report = obs.report()
        assert report.succeeded
        assert report.total_attempts == 1
        assert report.total_input_tokens == 10
        assert report.total_output_tokens == 5
        assert report.tokens_used["total"] == 15

    def test_record_failure_then_success(self):
        obs = ExtractObserver()
        failure = ValidationFailure("age", "below min", -5)
        obs.record_attempt(
            attempt_number=1,
            provider="openai",
            model="m",
            raw_response='{"age": -5}',
            parsed_data={"age": -5},
            validation_failures=[failure],
            input_tokens=10,
            output_tokens=5,
            latency_ms=150.0,
            succeeded=False,
        )
        obs.record_attempt(
            attempt_number=2,
            provider="openai",
            model="m",
            raw_response='{"age": 30}',
            parsed_data={"age": 30},
            validation_failures=[],
            input_tokens=15,
            output_tokens=5,
            latency_ms=120.0,
            succeeded=True,
        )
        report = obs.report()
        assert report.succeeded
        assert report.total_attempts == 2
        assert report.winning_attempt == 2
        assert len(report.validation_failures) == 1

    def test_reset_clears_state(self):
        obs = ExtractObserver()
        obs.record_attempt(1, "p", "m", "{}", {}, [], 0, 0, 0, True)
        obs.reset()
        assert obs.report().total_attempts == 0

    def test_summary_string(self):
        obs = ExtractObserver()
        obs.record_attempt(1, "openai", "gpt-4o-mini", '{"x":1}', {"x": 1}, [], 5, 3, 100.0, True)
        summary = obs.report().summary()
        assert "openai" in summary
        assert "OK" in summary


# ---------------------------------------------------------------------------
# _run_extraction with MockAdapter
# ---------------------------------------------------------------------------

class TestRunExtractionMock:
    def test_first_attempt_success(self):
        schema = Schema({"name": str, "age": int})
        adapter = MockAdapter(['{"name": "Alice", "age": 30}'])
        result = _run_extraction(adapter, [], schema, max_retries=3, observer=None,
                                  provider_name="mock", model_name="mock")
        assert result.succeeded
        assert result.data == {"name": "Alice", "age": 30}
        assert result.attempts == 1

    def test_retry_on_invalid_json(self):
        schema = Schema({"x": int})
        adapter = MockAdapter(["not json at all", '{"x": 5}'])
        result = _run_extraction(adapter, [], schema, max_retries=3, observer=None,
                                  provider_name="mock", model_name="mock")
        assert result.succeeded
        assert result.data["x"] == 5
        assert result.attempts == 2

    def test_retry_on_semantic_failure(self):
        schema = Schema({"age": int})
        schema.add_rule(SemanticRule("age", min_value=0))
        adapter = MockAdapter(['{"age": -5}', '{"age": 25}'])
        result = _run_extraction(adapter, [], schema, max_retries=3, observer=None,
                                  provider_name="mock", model_name="mock")
        assert result.succeeded
        assert result.data["age"] == 25

    def test_all_retries_exhausted(self):
        schema = Schema({"age": int})
        schema.add_rule(SemanticRule("age", min_value=0))
        adapter = MockAdapter(['{"age": -1}', '{"age": -2}', '{"age": -3}'])
        result = _run_extraction(adapter, [], schema, max_retries=3, observer=None,
                                  provider_name="mock", model_name="mock")
        assert not result.succeeded
        assert result.attempts == 3
        assert len(result.failures) > 0

    def test_observer_records_attempts(self):
        schema = Schema({"x": int})
        schema.add_rule(SemanticRule("x", min_value=0))
        obs = ExtractObserver()
        adapter = MockAdapter(['{"x": -1}', '{"x": 10}'])
        result = _run_extraction(adapter, [], schema, max_retries=3, observer=obs,
                                  provider_name="mock", model_name="mock")
        report = obs.report()
        assert result.succeeded
        assert report.total_attempts == 2
        assert report.winning_attempt == 2

    def test_type_coercion_applied(self):
        schema = Schema({"count": int})
        adapter = MockAdapter(['{"count": "7"}'])
        result = _run_extraction(adapter, [], schema, max_retries=1, observer=None,
                                  provider_name="mock", model_name="mock")
        assert result.succeeded
        assert result.data["count"] == 7

    def test_code_fence_stripped(self):
        schema = Schema({"val": int})
        adapter = MockAdapter(['```json\n{"val": 42}\n```'])
        result = _run_extraction(adapter, [], schema, max_retries=1, observer=None,
                                  provider_name="mock", model_name="mock")
        assert result.succeeded
        assert result.data["val"] == 42


# ---------------------------------------------------------------------------
# ExtractValidationError
# ---------------------------------------------------------------------------

class TestExtractValidationError:
    def test_raised_when_all_fail(self):
        schema = Schema({"age": int})
        schema.add_rule(SemanticRule("age", min_value=0))
        adapter = MockAdapter(['{"age": -1}', '{"age": -2}', '{"age": -3}'])

        from llm_extract.extractor import _run_extraction
        result = _run_extraction(adapter, [], schema, 3, None, "mock", "mock")

        with pytest.raises(ExtractValidationError) as exc_info:
            raise ExtractValidationError(result)

        assert exc_info.value.result is result
        assert "attempts" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

class TestMessageBuilders:
    def test_initial_messages_has_system_and_user(self):
        schema = Schema({"x": int})
        msgs = _build_initial_messages("extract x from: x is 5", schema)
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles

    def test_system_contains_schema(self):
        schema = Schema({"value": float})
        msgs = _build_initial_messages("test", schema)
        system_content = next(m["content"] for m in msgs if m["role"] == "system")
        assert "value" in system_content

    def test_system_hint_appended(self):
        schema = Schema({"x": int})
        msgs = _build_initial_messages("test", schema, system_hint="Always use metric units.")
        system_content = next(m["content"] for m in msgs if m["role"] == "system")
        assert "metric units" in system_content

    def test_retry_messages_include_correction(self):
        original = [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "extract"},
        ]
        failures = [ValidationFailure("age", "below minimum 0", -5)]
        retry = _build_retry_messages(original, '{"age": -5}', failures)
        roles = [m["role"] for m in retry]
        assert retry[-1]["role"] == "user"
        assert "age" in retry[-1]["content"]
