"""Grounded answer generation: send RAGRAM prompts to trending LLM backends.

Three backends share one ABC (:class:`GroundedGenerator`):

* :class:`OpenAIGenerator`     -- OpenAI chat completions        (extra ``[openai]``)
* :class:`AnthropicGenerator`  -- Anthropic Messages API         (extra ``[anthropic]``)
* :class:`OllamaGenerator`     -- local Ollama server            (stdlib only, zero deps)

Every backend consumes the exact ``system``/``user`` messages produced by
``RagramFusionEngine.build_prompt()`` (or ``GenerationPrompt.to_messages()``),
so answers stay grounded in the reranked evidence and the model's citation
policy travels with the prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prompts import SYSTEM_PROMPT, GenerationPrompt

logger = logging.getLogger("ragram.integrations.generation")

__all__ = [
    "AnthropicGenerator",
    "GenerationError",
    "GenerationResult",
    "GroundedGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "normalize_messages",
]

Message = dict[str, str]

_ROLES = frozenset({"system", "user", "assistant"})


class GenerationError(RuntimeError):
    """Raised when a generator is misconfigured or the LLM backend rejects a call."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A grounded answer plus enough context to audit how it was produced."""

    answer: str
    model: str
    messages: tuple[Message, ...]
    usage: Mapping[str, int] | None = None


def normalize_messages(
    prompt: GenerationPrompt | str | Sequence[Mapping[str, Any]],
) -> list[Message]:
    """Coerce prompt inputs into an OpenAI/Anthropic/Ollama-ready message list.

    Accepts a :class:`GenerationPrompt` (system + user), a bare question string
    (wrapped with RAGRAM's grounding ``SYSTEM_PROMPT``), or an explicit
    sequence of ``{"role", "content"}`` mappings.
    """
    if isinstance(prompt, GenerationPrompt):
        raw = prompt.to_messages()
    elif isinstance(prompt, str):
        raw = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    else:
        raw = list(prompt)
    messages: list[Message] = []
    for item in raw:
        role = str(item.get("role", ""))
        content = item.get("content")
        if role not in _ROLES or not isinstance(content, str):
            raise GenerationError(f"invalid chat message: {item!r}")
        messages.append({"role": role, "content": content})
    if not messages:
        raise GenerationError("prompt produced zero messages")
    return messages


class GroundedGenerator(ABC):
    """Turns a retrieval-grounded prompt into an LLM answer."""

    @abstractmethod
    async def generate(
        self, prompt: GenerationPrompt | str | Sequence[Mapping[str, Any]]
    ) -> GenerationResult:
        """Send the prompt to the backend and return the answer."""

    async def answer(
        self,
        engine: Any,
        query: str,
        *,
        top_n: int | None = None,
        relevance_threshold: float | None = None,
    ) -> GenerationResult:
        """One-call pipeline: hybrid search -> prompt -> grounded generation."""
        prompt = await engine.build_prompt(
            query, top_n=top_n, relevance_threshold=relevance_threshold
        )
        return await self.generate(prompt)

    async def aclose(self) -> None:
        """Release backend clients. Default is a no-op."""
        return None


class OpenAIGenerator(GroundedGenerator):
    """OpenAI chat completions (``gpt-4o-mini`` by default)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise GenerationError(
                "openai is not installed. Install with: pip install 'ragram-pro[openai]'"
            ) from exc
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise GenerationError(
                f"OpenAI API key missing: pass api_key=... or set ${api_key_env}"
            )
        self._model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self._temperature = temperature
        self._max_tokens = max_tokens
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)

    async def generate(self, prompt: GenerationPrompt | str | Sequence[Mapping[str, Any]]) -> GenerationResult:
        messages = normalize_messages(prompt)
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as exc:
            raise GenerationError(f"OpenAI request failed: {exc}") from exc
        if not response.choices or response.choices[0].message.content is None:
            raise GenerationError("OpenAI returned no message content")
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": int(response.usage.prompt_tokens or 0),
                "completion_tokens": int(response.usage.completion_tokens or 0),
            }
        logger.info("openai generation complete: model=%s", response.model)
        return GenerationResult(
            answer=response.choices[0].message.content,
            model=response.model,
            messages=tuple(messages),
            usage=usage,
        )

    async def aclose(self) -> None:
        await self._client.close()


class AnthropicGenerator(GroundedGenerator):
    """Anthropic Messages API (``claude-sonnet-4-5`` by default).

    The system message is lifted out of the message list because the
    Anthropic API takes ``system`` as a dedicated parameter.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise GenerationError(
                "anthropic is not installed. Install with: pip install 'ragram-pro[anthropic]'"
            ) from exc
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise GenerationError(
                f"Anthropic API key missing: pass api_key=... or set ${api_key_env}"
            )
        self._model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=key, timeout=timeout)

    async def generate(self, prompt: GenerationPrompt | str | Sequence[Mapping[str, Any]]) -> GenerationResult:
        messages = normalize_messages(prompt)
        system_text = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        dialogue = [m for m in messages if m["role"] != "system"]
        if not dialogue:
            raise GenerationError("prompt has no user/assistant messages for Anthropic")
        request: dict[str, Any] = {
            "model": self._model,
            "messages": dialogue,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if system_text:
            request["system"] = system_text
        try:
            response = await self._client.messages.create(**request)
        except Exception as exc:
            raise GenerationError(f"Anthropic request failed: {exc}") from exc
        answer = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        if not answer:
            raise GenerationError("Anthropic returned no text content")
        usage = {
            "prompt_tokens": int(getattr(response.usage, "input_tokens", 0)),
            "completion_tokens": int(getattr(response.usage, "output_tokens", 0)),
        }
        logger.info("anthropic generation complete: model=%s", self._model)
        return GenerationResult(
            answer=answer, model=self._model, messages=tuple(messages), usage=usage
        )

    async def aclose(self) -> None:
        await self._client.close()


class OllamaGenerator(GroundedGenerator):
    """Local Ollama server via its ``/api/chat`` REST endpoint.

    Uses only the Python standard library -- no SDK, no API key, works fully
    offline (e.g. ``ollama pull llama3.1`` then ``ollama serve``).
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float = 120.0,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL") or "llama3.1"
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._timeout = timeout
        self._options = dict(options or {})

    async def generate(self, prompt: GenerationPrompt | str | Sequence[Mapping[str, Any]]) -> GenerationResult:
        messages = normalize_messages(prompt)
        payload = {"model": self._model, "messages": messages, "stream": False}
        if self._options:
            payload["options"] = dict(self._options)
        try:
            body = await asyncio.to_thread(self._post, json.dumps(payload).encode("utf-8"))
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        try:
            answer = str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"unexpected Ollama response: {body!r}") from exc
        if not answer:
            raise GenerationError("Ollama returned an empty message")
        usage = None
        if "prompt_eval_count" in body or "eval_count" in body:
            usage = {
                "prompt_tokens": int(body.get("prompt_eval_count", 0)),
                "completion_tokens": int(body.get("eval_count", 0)),
            }
        logger.info("ollama generation complete: model=%s", self._model)
        return GenerationResult(
            answer=answer, model=str(body.get("model", self._model)), messages=tuple(messages), usage=usage
        )

    def _post(self, data: bytes) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise GenerationError(
                f"Ollama unreachable at {self._host} ({exc.reason}). "
                "Is the server running? try: ollama serve"
            ) from exc
