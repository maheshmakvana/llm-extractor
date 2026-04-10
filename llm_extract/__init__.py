"""
llm-extractor
===========
Extract structured, validated JSON from any LLM.

    pip install llm-extractor

Quick start::

    from llm_extract import extract, Schema, SemanticRule

    schema = Schema({"name": str, "age": int})
    schema.add_rule(SemanticRule("age", min_value=0, max_value=150))

    result = extract(
        prompt="John Doe is 34 years old.",
        schema=schema,
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-...",
    )
    print(result.data)  # {'name': 'John Doe', 'age': 34}
"""

from .core import (
    Schema,
    SemanticRule,
    SchemaField,
    ValidationFailure,
    SchemaInput,
)
from .extractor import (
    extract,
    aextract,
    ExtractResult,
    ExtractValidationError,
)
from .observability import (
    ExtractObserver,
    ForgeReport as ExtractReport,
    AttemptRecord,
)
from .providers import (
    ProviderConfig,
    ProviderResponse,
)

from .advanced import (
    ExtractionCache,
    RateLimiter,
    batch_extract,
    abatch_extract,
    ConfidenceScorer,
    SchemaEvolver,
    ExtractionPipeline,
    extract_with_budget,
)

__version__ = "1.1.0"
__author__ = "Mahesh Makvana"
__all__ = [
    # Core
    "Schema",
    "SemanticRule",
    "SchemaField",
    "ValidationFailure",
    "SchemaInput",
    # Extraction
    "extract",
    "aextract",
    "ExtractResult",
    "ExtractValidationError",
    # Observability
    "ExtractObserver",
    "ExtractReport",
    "AttemptRecord",
    # Provider config (for advanced use)
    "ProviderConfig",
    "ProviderResponse",
    # Advanced (1.1.0)
    "ExtractionCache",
    "RateLimiter",
    "batch_extract",
    "abatch_extract",
    "ConfidenceScorer",
    "SchemaEvolver",
    "ExtractionPipeline",
    "extract_with_budget",
]
