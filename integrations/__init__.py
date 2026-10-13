"""Optional integrations with trending RAG/LLM tooling.

Import style::

    from integrations import OpenAIGenerator, OllamaGenerator
    from integrations import RagramRetriever              # LangChain (lazy)
    from integrations import RagramLlamaIndexRetriever    # LlamaIndex (lazy)
    from integrations.langchain import RagramRetriever    # also fine

Framework-backed names are resolved lazily (PEP 562): importing this package
never pulls in LangChain or LlamaIndex, and touching a missing framework
raises an ``ImportError`` that names the exact install extra
(``ragram-pro[langchain]`` / ``ragram-pro[llamaindex]``).
Generation backends import their SDKs inside the constructors, so this module
stays dependency-free.
"""

from __future__ import annotations

from importlib import import_module

from integrations.generation import (
    AnthropicGenerator,
    GenerationError,
    GenerationResult,
    GroundedGenerator,
    OllamaGenerator,
    OpenAIGenerator,
    normalize_messages,
)

_LAZY_ATTRS: dict[str, str] = {
    "RagramRetriever": "integrations.langchain",
    "RagramLlamaIndexRetriever": "integrations.llamaindex",
}

__all__ = [
    "AnthropicGenerator",
    "GenerationError",
    "GenerationResult",
    "GroundedGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "RagramLlamaIndexRetriever",
    "RagramRetriever",
    "normalize_messages",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name]), name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
