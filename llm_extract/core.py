"""
core.py — Schema definition and type coercion for llm-extractor.

Supports:
  - dict-based schemas  {"field": type, ...}
  - pydantic BaseModel subclasses
  - raw JSON Schema dicts  {"type": "object", "properties": {...}}
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

_PYDANTIC_AVAILABLE = False
try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None  # type: ignore
    PydanticValidationError = None  # type: ignore

_JSONSCHEMA_AVAILABLE = False
try:
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

SchemaInput = Union[
    Dict[str, Any],           # simple {"field": type} or raw JSON Schema
    Type,                     # pydantic BaseModel subclass
]


class SchemaField:
    """Metadata for a single field derived from the schema."""

    def __init__(
        self,
        name: str,
        python_type: type,
        required: bool = True,
        default: Any = None,
    ) -> None:
        self.name = name
        self.python_type = python_type
        self.required = required
        self.default = default

    def __repr__(self) -> str:
        return f"SchemaField({self.name!r}, {self.python_type.__name__}, required={self.required})"


class SemanticRule:
    """
    A semantic validation rule attached to a field.

    Parameters
    ----------
    field : str
        Name of the field this rule applies to.
    min_value : numeric, optional
        Minimum allowed numeric value (inclusive).
    max_value : numeric, optional
        Maximum allowed numeric value (inclusive).
    allowed_values : list, optional
        Whitelist of acceptable values.
    pattern : str, optional
        Regex pattern the string value must match.
    validator : callable, optional
        ``f(value) -> bool`` — custom predicate; return False to fail.
    message : str, optional
        Human-readable failure message (used in correction prompts).
    """

    def __init__(
        self,
        field: str,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[List[Any]] = None,
        pattern: Optional[str] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values
        self.pattern = re.compile(pattern) if pattern else None
        self._pattern_str = pattern
        self.validator = validator
        self.message = message

    def check(self, value: Any) -> Tuple[bool, str]:
        """
        Return ``(ok, reason)`` where ``ok`` is True when the value passes.
        """
        if value is None:
            return True, ""

        if self.min_value is not None:
            try:
                if float(value) < self.min_value:
                    return False, (
                        self.message
                        or f"'{self.field}' value {value!r} is below minimum {self.min_value}"
                    )
            except (TypeError, ValueError):
                return False, f"'{self.field}' is not numeric, cannot apply min_value"

        if self.max_value is not None:
            try:
                if float(value) > self.max_value:
                    return False, (
                        self.message
                        or f"'{self.field}' value {value!r} exceeds maximum {self.max_value}"
                    )
            except (TypeError, ValueError):
                return False, f"'{self.field}' is not numeric, cannot apply max_value"

        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, (
                    self.message
                    or f"'{self.field}' value {value!r} not in allowed values {self.allowed_values}"
                )

        if self.pattern is not None:
            if not isinstance(value, str):
                return False, f"'{self.field}' must be a string to match pattern"
            if not self.pattern.search(str(value)):
                return False, (
                    self.message
                    or f"'{self.field}' value {value!r} does not match pattern {self._pattern_str!r}"
                )

        if self.validator is not None:
            try:
                ok = self.validator(value)
            except Exception as exc:
                return False, f"'{self.field}' custom validator raised: {exc}"
            if not ok:
                return False, (
                    self.message or f"'{self.field}' value {value!r} failed custom validation"
                )

        return True, ""


class ValidationFailure:
    """Records a single field-level validation failure."""

    def __init__(self, field: str, reason: str, value: Any = None) -> None:
        self.field = field
        self.reason = reason
        self.value = value

    def __repr__(self) -> str:
        return f"ValidationFailure(field={self.field!r}, reason={self.reason!r})"


class Schema:
    """
    Unified schema wrapper.

    Accepts:
      - ``{"field": type, ...}``          simple type map
      - a pydantic ``BaseModel`` subclass
      - ``{"type": "object", ...}``        raw JSON Schema

    Usage::

        schema = Schema({"name": str, "age": int})
        schema.add_rule(SemanticRule("age", min_value=0, max_value=150))
    """

    def __init__(self, definition: SchemaInput) -> None:
        self._definition = definition
        self._rules: List[SemanticRule] = []
        self._fields: List[SchemaField] = []
        self._pydantic_model: Optional[type] = None
        self._json_schema: Optional[Dict[str, Any]] = None

        self._parse_definition(definition)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_rule(self, rule: SemanticRule) -> "Schema":
        """Attach a semantic rule. Returns self for chaining."""
        self._rules.append(rule)
        return self

    @property
    def fields(self) -> List[SchemaField]:
        return list(self._fields)

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self._fields]

    def to_json_schema(self) -> Dict[str, Any]:
        """Return a JSON Schema dict describing the expected output object."""
        if self._json_schema:
            return self._json_schema

        if self._pydantic_model is not None and _PYDANTIC_AVAILABLE:
            return self._pydantic_model.model_json_schema()

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for field in self._fields:
            prop = _python_type_to_json_schema(field.python_type)
            properties[field.name] = prop
            if field.required:
                required.append(field.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_prompt_description(self) -> str:
        """Human-readable description of expected output, for system prompts."""
        js = self.to_json_schema()
        return (
            "Respond with a valid JSON object matching this schema:\n"
            + json.dumps(js, indent=2)
        )

    def validate_data(self, data: Dict[str, Any]) -> List[ValidationFailure]:
        """
        Full validation pass: structural (type-check) + semantic rules.

        Returns a list of ``ValidationFailure`` objects; empty list = valid.
        """
        failures: List[ValidationFailure] = []

        # --- structural ---
        failures.extend(self._structural_validate(data))

        # --- semantic rules ---
        for rule in self._rules:
            if rule.field in data:
                ok, reason = rule.check(data[rule.field])
                if not ok:
                    failures.append(ValidationFailure(rule.field, reason, data[rule.field]))

        return failures

    def coerce(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort type coercion (e.g. "42" → 42 for int fields).
        Does not raise; problematic fields are left as-is for the validator to catch.
        """
        if self._pydantic_model is not None:
            return data  # pydantic handles coercion itself

        result = dict(data)
        for field in self._fields:
            if field.name in result:
                result[field.name] = _coerce_value(result[field.name], field.python_type)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_definition(self, definition: SchemaInput) -> None:
        if _PYDANTIC_AVAILABLE and isinstance(definition, type) and issubclass(definition, BaseModel):
            self._pydantic_model = definition
            for fname, finfo in definition.model_fields.items():
                python_type = finfo.annotation or Any
                required = finfo.is_required()
                default = finfo.default if not required else None
                self._fields.append(SchemaField(fname, python_type, required, default))
            return

        if isinstance(definition, dict):
            # Check if it looks like a raw JSON Schema
            if "type" in definition and definition.get("type") == "object" and "properties" in definition:
                self._json_schema = definition
                props = definition.get("properties", {})
                req_list = definition.get("required", list(props.keys()))
                for fname, prop in props.items():
                    python_type = _json_schema_type_to_python(prop)
                    required = fname in req_list
                    self._fields.append(SchemaField(fname, python_type, required))
                return

            # Simple type map: {"field": type}
            for fname, ftype in definition.items():
                if not isinstance(ftype, type):
                    raise ValueError(
                        f"Schema field '{fname}' must map to a Python type (e.g. str, int, float, bool, list, dict). "
                        f"Got: {ftype!r}"
                    )
                self._fields.append(SchemaField(fname, ftype, required=True))
            return

        raise TypeError(
            f"Unsupported schema definition type: {type(definition).__name__}. "
            "Pass a dict {'field': type}, a pydantic BaseModel class, or a JSON Schema dict."
        )

    def _structural_validate(self, data: Dict[str, Any]) -> List[ValidationFailure]:
        failures = []

        if self._pydantic_model is not None and _PYDANTIC_AVAILABLE:
            try:
                self._pydantic_model(**data)
            except PydanticValidationError as exc:
                for err in exc.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    failures.append(ValidationFailure(loc, err["msg"], data.get(loc)))
            return failures

        if self._json_schema and _JSONSCHEMA_AVAILABLE:
            validator = jsonschema.Draft7Validator(self._json_schema)
            for err in validator.iter_errors(data):
                field = err.path[-1] if err.path else str(err.schema_path)
                failures.append(ValidationFailure(str(field), err.message))
            return failures

        # Simple type-map validation
        for field in self._fields:
            if field.name not in data:
                if field.required:
                    failures.append(ValidationFailure(field.name, f"required field '{field.name}' is missing"))
                continue
            val = data[field.name]
            if not _isinstance_loose(val, field.python_type):
                failures.append(
                    ValidationFailure(
                        field.name,
                        f"'{field.name}' expected {field.python_type.__name__}, got {type(val).__name__}",
                        val,
                    )
                )

        return failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_type_to_json_schema(python_type: type) -> Dict[str, Any]:
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return mapping.get(python_type, {"type": "string"})


def _json_schema_type_to_python(prop: Dict[str, Any]) -> type:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(prop.get("type", "string"), str)


def _coerce_value(value: Any, target_type: type) -> Any:
    if value is None:
        return value
    try:
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if target_type == int:
            return int(float(value))
        if target_type == float:
            return float(value)
        if target_type == str:
            return str(value)
    except (ValueError, TypeError):
        pass
    return value


def _isinstance_loose(value: Any, expected_type: type) -> bool:
    """bool is a subclass of int in Python, handle that gracefully."""
    if expected_type == float and isinstance(value, int):
        return True
    if expected_type == int and isinstance(value, bool):
        return False  # booleans should not satisfy int fields
    return isinstance(value, expected_type)


def build_correction_prompt(failures: List[ValidationFailure], raw_response: str) -> str:
    """
    Build a correction prompt that tells the LLM exactly what went wrong.
    """
    lines = [
        "Your previous response had the following validation errors.",
        "Please return a corrected JSON object that fixes all issues.",
        "",
        "Errors found:",
    ]
    for i, failure in enumerate(failures, 1):
        lines.append(f"  {i}. Field '{failure.field}': {failure.reason}")
    lines += [
        "",
        f"Your previous (invalid) response was:\n{raw_response}",
        "",
        "Return ONLY the corrected JSON object with no extra text, markdown, or code fences.",
    ]
    return "\n".join(lines)
