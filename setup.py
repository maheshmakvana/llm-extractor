from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-extractor",
    version="1.0.0",
    author="Mahesh Makvana",
    description=(
        "Extract structured, validated JSON from any LLM — "
        "OpenAI, Anthropic, Gemini — with schema validation, semantic rules, and auto-retry."
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
