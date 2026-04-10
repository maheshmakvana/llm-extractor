from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-extractor",
    version="1.2.0",
    author="Mahesh Makvana",
    description=(
        "Extract structured, validated JSON from any LLM — "
        "OpenAI, Anthropic, Gemini — with batch extraction, caching, per-field confidence scoring, "
        "schema evolution, multi-schema extraction, output transforms, partial extraction, "
        "extraction diff, pipeline extraction, and smart auto-retry."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maheshmakvana/llm-extractor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    keywords=[
        "llm extraction",
        "structured output",
        "json extraction",
        "llm json",
        "openai structured output",
        "anthropic structured output",
        "pydantic extraction",
        "schema validation llm",
        "auto retry llm",
        "batch llm extraction",
        "llm caching",
        "confidence scoring llm",
        "schema migration",
        "llm pipeline",
        "extract json from llm",
        "llm schema",
        "langchain extraction",
        "openai function calling",
        "llm output parsing",
        "ai data extraction",
        "structured ai output",
        "rate limiter llm",
        "async llm extraction",
        "partial extraction llm",
        "multi schema llm",
        "output transformer llm",
        "extraction diff",
        "field confidence llm",
        "llm structured data",
        "gpt4 json extraction",
        "claude json extraction",
        "gemini structured output",
        "llm output validation",
        "extract data from text ai",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
        "jsonschema>=4.0",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.20"],
        "google": ["google-generativeai>=0.5"],
        "all": ["openai>=1.0", "anthropic>=0.20", "google-generativeai>=0.5"],
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21", "black", "isort"],
    },
)
