# llm-extractor

**Extract structured, validated JSON from any LLM.**

`pip install llm-extractor` — then stop fighting JSON parsing bugs, provider-specific APIs, and silent semantic failures. One unified interface to extract structured data from OpenAI, Anthropic, and Gemini — with automatic retries, semantic rules, and full observability.

## The Problem (2026)

Even with native structured outputs, Python developers still hit:

| Pain | Reality |
|------|---------|
| Provider fragmentation | OpenAI, Anthropic, Gemini all use different structured output APIs |
| Semantic failures | Valid JSON with nonsense values (`price: -999`, `email: "not-an-email"`) |
| Silent failures | Model returns `{}` or truncated object — no error raised |
| Dumb retries | Most code retries blindly with the same broken prompt |
| Zero observability | You know it failed but not *why* or *how often* |

`llm-extractor` fixes all five.

## Installation

```bash
pip install llm-extractor                   # core only
pip install "llm-extractor[openai]"         # + OpenAI
pip install "llm-extractor[anthropic]"      # + Anthropic
pip install "llm-extractor[google]"         # + Gemini
pip install "llm-extractor[all]"            # all providers
```

## Quick Start

```python
from llm_extract import extract, Schema, SemanticRule

# 1. Define your output schema
schema = Schema({
    "name": str,
    "age": int,
    "email": str,
    "score": float,
})

# 2. Add semantic rules
schema.add_rule(SemanticRule("age", min_value=0, max_value=150))
schema.add_rule(SemanticRule("score", min_value=0.0, max_value=100.0))
schema.add_rule(SemanticRule("email", pattern=r"^[^@]+@[^@]+\.[^@]+$"))

# 3. Extract structured output — works across all providers
result = extract(
    prompt="Extract info: John Doe, 34 years old, john@example.com, scored 87.5",
    schema=schema,
    provider="openai",          # or "anthropic", "gemini", "auto"
    model="gpt-4o-mini",
    api_key="sk-...",
    max_retries=3,
)

print(result.data)
# {'name': 'John Doe', 'age': 34, 'email': 'john@example.com', 'score': 87.5}

print(result.attempts)   # 1
print(result.provider)   # 'openai'
```

## Pydantic Models

```python
from pydantic import BaseModel
from llm_extract import extract

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool
    tags: list[str]

result = extract(
    prompt="Extract: Blue Widget, costs $29.99, currently available, tagged as gadget and home",
    schema=Product,
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    api_key="sk-ant-...",
)

product: Product = result.typed_data(Product)
print(product.price)  # 29.99
```

## Semantic Rules

```python
from llm_extract import SemanticRule, Schema

schema = Schema({"status": str, "count": int, "ratio": float})

# Enum constraint
schema.add_rule(SemanticRule("status", allowed_values=["active", "inactive", "pending"]))

# Range constraint
schema.add_rule(SemanticRule("count", min_value=0))
schema.add_rule(SemanticRule("ratio", min_value=0.0, max_value=1.0))

# Regex pattern
schema.add_rule(SemanticRule("email", pattern=r"^[^@]+@[^@]+\.[^@]+$"))

# Custom validator function
schema.add_rule(SemanticRule("count", validator=lambda v: v % 2 == 0, message="count must be even"))
```

## Observability

```python
from llm_extract import extract, ExtractObserver

observer = ExtractObserver()

result = extract(
    prompt="...",
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    api_key="...",
    observer=observer,
)

# Per-call report
report = observer.report()
print(report.total_attempts)       # 2
print(report.validation_failures)  # [ValidationFailure(field='age', reason='below min_value 0')]
print(report.raw_responses)        # ['{"age": -5, ...}', '{"age": 34, ...}']
print(report.latency_ms)           # [342, 289]
print(report.tokens_used)          # {'input': 120, 'output': 45}
```

## Multi-Provider Fallback

```python
result = extract(
    prompt="...",
    schema=schema,
    provider="auto",   # tries providers in priority order
    fallback_chain=[
        {"provider": "openai",    "model": "gpt-4o-mini",               "api_key": "sk-..."},
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",  "api_key": "sk-ant-..."},
        {"provider": "gemini",    "model": "gemini-1.5-flash",           "api_key": "AIza..."},
    ],
    max_retries=2,
)
print(result.provider)  # whichever succeeded
```

## Async Support

```python
import asyncio
from llm_extract import aextract

async def main():
    result = await aextract(
        prompt="...",
        schema=schema,
        provider="openai",
        model="gpt-4o-mini",
        api_key="...",
    )
    print(result.data)

asyncio.run(main())
```

## Raise on Failure

```python
from llm_extract import extract, ExtractValidationError

try:
    result = extract(..., raise_on_failure=True)
except ExtractValidationError as e:
    print(e.result.failures)   # list of ValidationFailure
    print(e.result.raw)        # last raw LLM response
```

## JSON Schema Input

```python
from llm_extract import extract, Schema

schema = Schema({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year":  {"type": "integer"},
        "rating": {"type": "number"}
    },
    "required": ["title", "year", "rating"]
})

result = extract(prompt="...", schema=schema, ...)
```

## OpenAI-Compatible Endpoints

```python
result = extract(
    prompt="...",
    schema=schema,
    provider="openai",
    model="mistral-7b-instruct",
    api_key="your-key",
    base_url="https://your-openai-compatible-endpoint/v1",
)
```

## Why llm-extractor?

- **Unified API** — one interface for OpenAI, Anthropic, Gemini, and any OpenAI-compatible endpoint
- **Schema-first** — define once with `dict`, `pydantic.BaseModel`, or JSON Schema
- **Semantic rules** — enforce business logic, not just types
- **Smart retries** — correction prompts tell the model *exactly* what went wrong
- **Full observability** — every attempt, failure, token count, and latency recorded
- **Zero magic** — no hidden prompt injection, no global state, fully inspectable

## License

MIT
