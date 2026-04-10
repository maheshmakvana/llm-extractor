"""
providers.py — Unified adapter layer for OpenAI, Anthropic, Gemini,
and any OpenAI-compatible endpoint.

Each adapter implements ``ProviderAdapter.complete()`` which returns a
``ProviderResponse`` with the raw text and token counts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    raw: Any = None  # original SDK response object


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None          # for OpenAI-compatible endpoints
    timeout: float = 30.0
    temperature: float = 0.0
    max_output_tokens: int = 1024
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class ProviderAdapter:
    """Abstract base for all provider adapters."""

    def complete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        raise NotImplementedError

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI adapter (also used for OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

class OpenAIAdapter(ProviderAdapter):
    """
    Supports:
    - OpenAI native (gpt-4o, gpt-4o-mini, o1, …)
    - Any OpenAI-compatible endpoint via ``base_url``
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for the OpenAI provider. "
                    "Install with: pip install 'llm-extract[openai]'"
                )
            kwargs: Dict[str, Any] = {
                "api_key": self._config.api_key,
                "timeout": self._config.timeout,
            }
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        client = self._get_client()
        start = time.monotonic()

        call_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
            **self._config.extra_kwargs,
        }

        if json_schema is not None:
            # Use native structured output when schema provided
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        else:
            call_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**call_kwargs)
        latency = (time.monotonic() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage

        return ProviderResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            provider="openai",
            model=self._config.model,
            raw=response,
        )

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install 'llm-extract[openai]'"
            )

        kwargs: Dict[str, Any] = {
            "api_key": self._config.api_key,
            "timeout": self._config.timeout,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        async_client = AsyncOpenAI(**kwargs)

        start = time.monotonic()

        call_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
            **self._config.extra_kwargs,
        }

        if json_schema is not None:
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        else:
            call_kwargs["response_format"] = {"type": "json_object"}

        response = await async_client.chat.completions.create(**call_kwargs)
        latency = (time.monotonic() - start) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage

        return ProviderResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
            provider="openai",
            model=self._config.model,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """Supports claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-*, etc."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for the Anthropic provider. "
                    "Install with: pip install 'llm-extract[anthropic]'"
                )
            self._client = anthropic.Anthropic(
                api_key=self._config.api_key,
                timeout=self._config.timeout,
            )
        return self._client

    def _build_tool_for_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap the JSON schema as an Anthropic tool_use tool."""
        return {
            "name": "structured_output",
            "description": "Extract structured data according to the provided schema.",
            "input_schema": json_schema,
        }

    def complete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        client = self._get_client()
        start = time.monotonic()

        # Anthropic uses separate system param
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_text = "\n".join(system_parts) if system_parts else None

        call_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": user_messages,
            "max_tokens": self._config.max_output_tokens,
            **self._config.extra_kwargs,
        }
        if system_text:
            call_kwargs["system"] = system_text

        if json_schema is not None:
            tool = self._build_tool_for_schema(json_schema)
            call_kwargs["tools"] = [tool]
            call_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        response = client.messages.create(**call_kwargs)
        latency = (time.monotonic() - start) * 1000

        # Extract text or tool_use block
        text = ""
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    text = json.dumps(block.input)
                    break
                elif block.type == "text":
                    text = block.text

        usage = response.usage

        return ProviderResponse(
            text=text,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency,
            provider="anthropic",
            model=self._config.model,
            raw=response,
        )

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install 'llm-extract[anthropic]'"
            )

        async_client = anthropic.AsyncAnthropic(
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )
        start = time.monotonic()

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_text = "\n".join(system_parts) if system_parts else None

        call_kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": user_messages,
            "max_tokens": self._config.max_output_tokens,
            **self._config.extra_kwargs,
        }
        if system_text:
            call_kwargs["system"] = system_text

        if json_schema is not None:
            tool = self._build_tool_for_schema(json_schema)
            call_kwargs["tools"] = [tool]
            call_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        response = await async_client.messages.create(**call_kwargs)
        latency = (time.monotonic() - start) * 1000

        text = ""
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    text = json.dumps(block.input)
                    break
                elif block.type == "text":
                    text = block.text

        usage = response.usage

        return ProviderResponse(
            text=text,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency,
            provider="anthropic",
            model=self._config.model,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------

class GeminiAdapter(ProviderAdapter):
    """Supports Gemini 1.5 Flash, Gemini 2.0 Flash, Gemini Ultra, etc."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    def _get_model(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for the Gemini provider. "
                "Install with: pip install 'llm-extract[google]'"
            )
        genai.configure(api_key=self._config.api_key)
        generation_config = {
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "response_mime_type": "application/json",
        }
        return genai.GenerativeModel(
            model_name=self._config.model,
            generation_config=generation_config,
        )

    def _messages_to_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Flatten messages to a single prompt string for Gemini."""
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                parts.append(f"[System]: {content}")
            elif role == "USER":
                parts.append(f"[User]: {content}")
            elif role == "ASSISTANT":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def complete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        model = self._get_model()
        prompt = self._messages_to_gemini(messages)

        if json_schema is not None:
            prompt += (
                f"\n\nReturn ONLY a JSON object matching this schema:\n"
                + json.dumps(json_schema, indent=2)
            )

        start = time.monotonic()
        response = model.generate_content(prompt)
        latency = (time.monotonic() - start) * 1000

        text = response.text or ""

        # Gemini sometimes wraps in ```json ... ```
        text = _strip_code_fence(text)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            provider="gemini",
            model=self._config.model,
            raw=response,
        )

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        # google-generativeai async is a thin wrapper; run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, messages, json_schema)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "google": GeminiAdapter,
}


def get_adapter(config: ProviderConfig) -> ProviderAdapter:
    """Return the appropriate adapter for the given provider name."""
    provider = config.provider.lower()
    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        # Assume OpenAI-compatible if unknown — user should supply base_url
        cls = OpenAIAdapter
    return cls(config)


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences that some models wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (e.g. ```json or just ```)
        lines = text.splitlines()
        start = 1
        if lines[0].strip().startswith("```"):
            pass  # skip first line
        end = len(lines)
        if lines[-1].strip() == "```":
            end = len(lines) - 1
        text = "\n".join(lines[start:end]).strip()
    return text
