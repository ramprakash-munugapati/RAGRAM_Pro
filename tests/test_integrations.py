"""Integration suite for RAGRAM_Pro: lazy imports, generation backends,
LangChain / LlamaIndex retrievers, the FastAPI example and .env wiring.

Run with:  python tests/test_integrations.py
Framework-specific checks are skipped automatically when an optional extra
is not installed. No network access is required (Ollama is stubbed locally).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS += 1
    print(f"  ok: {name}")


def skip(name: str, why: str) -> None:
    print(f"  skip: {name} ({why})")


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Lazy imports (must run before any framework module is imported)             #
# --------------------------------------------------------------------------- #


def test_lazy_imports() -> None:
    print("[lazy imports]")
    import integrations

    check(
        "frameworks not eagerly imported",
        "langchain_core" not in sys.modules and "llama_index" not in sys.modules,
    )
    check("lazy retriever names exposed", "RagramRetriever" in dir(integrations))
    check(
        "generation backends importable eagerly",
        hasattr(integrations, "OllamaGenerator") and hasattr(integrations, "OpenAIGenerator"),
    )
    check(
        "unknown attribute raises AttributeError",
        _attr_error(integrations, "DoesNotExist"),
    )
    retriever = integrations.RagramRetriever  # resolves lazily
    check("lazy attribute resolves to class", retriever.__name__ == "RagramRetriever")


def _attr_error(module, name: str) -> bool:  # type: ignore[no-untyped-def]
    try:
        getattr(module, name)
    except AttributeError:
        return True
    return False


# --------------------------------------------------------------------------- #
# Generation: message normalization                                           #
# --------------------------------------------------------------------------- #


def test_normalize_messages() -> None:
    print("[generation: normalize_messages]")
    from integrations import GenerationError, normalize_messages
    from models import FusionResult, SearchStats
    from prompts import SYSTEM_PROMPT, build_generation_prompt

    messages = normalize_messages("what now?")
    check("string prompt wraps system+user", [m["role"] for m in messages] == ["system", "user"])
    check("string prompt uses grounding system", messages[0]["content"] == SYSTEM_PROMPT)

    result = FusionResult(query="q", chunks=(), stats=SearchStats(0, 0, 0, 0, 0, 0.0))
    prompt = build_generation_prompt("q", result)
    check("GenerationPrompt passthrough", normalize_messages(prompt) == prompt.to_messages())

    explicit = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    check("explicit messages passthrough", normalize_messages(explicit) == explicit)

    try:
        normalize_messages([{"role": "tool", "content": "x"}])
        rejected = False
    except GenerationError:
        rejected = True
    check("invalid role rejected", rejected)

    try:
        normalize_messages([])
        rejected = False
    except GenerationError:
        rejected = True
    check("empty message list rejected", rejected)


# --------------------------------------------------------------------------- #
# Generation: Ollama (stubbed local server, zero network)                     #
# --------------------------------------------------------------------------- #


class _OllamaStub(BaseHTTPRequestHandler):
    payload: dict | None = None

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        length = int(self.headers.get("Content-Length") or 0)
        _OllamaStub.payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "model": _OllamaStub.payload.get("model", "stub"),
                "message": {"role": "assistant", "content": "Grounded answer citing [C1]."},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 7,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence request logging
        del args


def _start_stub() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stub_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_ollama_generation() -> None:
    print("[generation: ollama]")
    from integrations import GenerationError, OllamaGenerator

    server = _start_stub()
    try:
        generator = OllamaGenerator(model="stub-model", host=_stub_url(server))
        result = asyncio.run(generator.generate("what is the escalation path?"))
        check("stub answer returned", result.answer.startswith("Grounded answer"))
        check("model echoed back", result.model == "stub-model")
        check("usage mapped", dict(result.usage or {}) == {"prompt_tokens": 5, "completion_tokens": 7})
        check(
            "messages sent to server",
            [m["role"] for m in _OllamaStub.payload["messages"]] == ["system", "user"],
        )
        check("streaming disabled", _OllamaStub.payload.get("stream") is False)
    finally:
        server.shutdown()

    dead = OllamaGenerator(model="stub-model", host="http://127.0.0.1:9", timeout=2)
    try:
        asyncio.run(dead.generate("hi"))
        unreachable = False
    except GenerationError as exc:
        unreachable = "ollama serve" in str(exc)
    check("unreachable server raises actionable error", unreachable)


# --------------------------------------------------------------------------- #
# Shared mini engine                                                          #
# --------------------------------------------------------------------------- #


def _make_engine():  # type: ignore[no-untyped-def]
    from config import FusionConfig
    from embedding import HashingEmbedder
    from engine import RagramFusionEngine
    from models import Chunk
    from rerank import TermOverlapReranker
    from storage import InMemoryStorageEngine

    chunks = [
        Chunk(
            id="c1",
            text="Payment outage escalation ladder: page the on-call payment engineer within five minutes.",
            metadata={"source": "runbooks/incident-response.md"},
        ),
        Chunk(
            id="c2",
            text="Refund approvals above 5000 USD require a finance manager sign-off.",
            metadata={"source": "policy/refunds.md"},
        ),
        Chunk(
            id="c3",
            text="A SEV1 is declared when checkout fails for more than ten minutes.",
            metadata={"source": "runbooks/slo.md"},
        ),
    ]
    engine = RagramFusionEngine(
        InMemoryStorageEngine(),
        HashingEmbedder(),
        TermOverlapReranker(),
        FusionConfig(top_n=3, relevance_threshold=0.0, candidate_limit=10),
    )
    asyncio.run(engine.initialize())
    asyncio.run(engine.index(chunks))
    return engine


def test_answer_pipeline(engine) -> None:  # type: ignore[no-untyped-def]
    print("[generation: engine.answer()]")
    from integrations import OllamaGenerator

    server = _start_stub()
    try:
        generator = OllamaGenerator(model="stub-model", host=_stub_url(server))
        result = asyncio.run(generator.answer(engine, "payment outage escalation"))
        user = next(m for m in result.messages if m["role"] == "user")
        check("answer() runs full search+prompt pipeline", "Retrieved Evidence" in user["content"])
        check("answer() returns backend answer", result.answer.startswith("Grounded answer"))
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# Generation: OpenAI / Anthropic constructors (no network calls)              #
# --------------------------------------------------------------------------- #


def _with_popped(keys: tuple[str, ...]):  # type: ignore[no-untyped-def]
    saved = {key: os.environ.pop(key, None) for key in keys}

    class _Ctx:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc: object) -> None:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return _Ctx()


def test_openai_generator() -> None:
    print("[generation: openai]")
    from integrations import GenerationError, OpenAIGenerator

    if not has("openai"):
        skip("openai constructor", "openai not installed; pip install 'ragram-pro[openai]'")
        return

    with _with_popped(("OPENAI_API_KEY", "OPENAI_MODEL")):
        try:
            OpenAIGenerator()
            raised = False
        except GenerationError as exc:
            raised = "OPENAI_API_KEY" in str(exc)
        check("missing key raises with env hint", raised)

        generator = OpenAIGenerator(api_key="test-key")
        check("default model fallback", generator._model == "gpt-4o-mini")

        os.environ["OPENAI_MODEL"] = "my-custom-model"
        check("model from env", OpenAIGenerator(api_key="k")._model == "my-custom-model")

    with _with_popped(("OPENAI_API_KEY", "OPENAI_BASE_URL")):
        os.environ["OPENAI_BASE_URL"] = "http://localhost:9/v1"
        generator = OpenAIGenerator(api_key="k")
        check("base_url from env", "localhost:9" in str(generator._client.base_url))


def test_anthropic_generator() -> None:
    print("[generation: anthropic]")
    from integrations import AnthropicGenerator, GenerationError

    if not has("anthropic"):
        skip("anthropic constructor", "anthropic not installed; pip install 'ragram-pro[anthropic]'")
        return

    with _with_popped(("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")):
        try:
            AnthropicGenerator()
            raised = False
        except GenerationError as exc:
            raised = "ANTHROPIC_API_KEY" in str(exc)
        check("missing key raises with env hint", raised)

        generator = AnthropicGenerator(api_key="test-key")
        check("default model fallback", generator._model == "claude-sonnet-4-5")

        os.environ["ANTHROPIC_MODEL"] = "my-claude"
        check("model from env", AnthropicGenerator(api_key="k")._model == "my-claude")


# --------------------------------------------------------------------------- #
# LangChain retriever                                                         #
# --------------------------------------------------------------------------- #


async def _ainvoke(retriever, query: str):  # type: ignore[no-untyped-def]
    return await retriever.ainvoke(query)


async def _invoke_inside_loop(retriever, query: str):  # type: ignore[no-untyped-def]
    return retriever.invoke(query)  # sync entry point called from async context


def test_langchain_retriever(engine) -> None:  # type: ignore[no-untyped-def]
    print("[langchain retriever]")
    if not has("langchain_core"):
        skip("RagramRetriever", "langchain-core not installed; pip install 'ragram-pro[langchain]'")
        return
    from integrations.langchain import RagramRetriever

    retriever = RagramRetriever(engine=engine, top_n=3, relevance_threshold=0.0)
    docs = retriever.invoke("payment outage escalation")
    check("sync invoke returns Documents", bool(docs) and all(d.page_content for d in docs))
    metadata = docs[0].metadata
    check(
        "metadata carries scores + streams",
        {"chunk_id", "relevance_score", "fusion_score", "streams"} <= set(metadata),
    )
    check("document id set", docs[0].id == metadata["chunk_id"])

    async_docs = asyncio.run(_ainvoke(retriever, "payment outage escalation"))
    check("async ainvoke returns docs", len(async_docs) == len(docs))

    nested = asyncio.run(_invoke_inside_loop(retriever, "payment outage escalation"))
    check("sync invoke inside running event loop", bool(nested))


# --------------------------------------------------------------------------- #
# LlamaIndex retriever                                                        #
# --------------------------------------------------------------------------- #


def test_llamaindex_retriever(engine) -> None:  # type: ignore[no-untyped-def]
    print("[llamaindex retriever]")
    if not has("llama_index.core"):
        skip(
            "RagramLlamaIndexRetriever",
            "llama-index-core not installed; pip install 'ragram-pro[llamaindex]'",
        )
        return
    from integrations.llamaindex import RagramLlamaIndexRetriever

    retriever = RagramLlamaIndexRetriever(engine, top_n=3, relevance_threshold=0.0)
    nodes = retriever.retrieve("payment outage escalation")
    check("retrieve returns NodeWithScore", bool(nodes) and all(n.score is not None for n in nodes))
    check("node text from chunk", all(n.text for n in nodes))
    check("node metadata carries scores", "relevance_score" in nodes[0].metadata)
    scores = [n.score for n in nodes]
    check("scores descending", scores == sorted(scores, reverse=True))

    other = retriever.retrieve("refund approval limit")
    check("second query works", bool(other))


# --------------------------------------------------------------------------- #
# FastAPI example                                                             #
# --------------------------------------------------------------------------- #


def test_fastapi_example() -> None:
    print("[fastapi example]")
    if not has("fastapi"):
        skip("examples/fastapi_app.py", "fastapi not installed; pip install 'ragram-pro[api]'")
        return
    from fastapi.testclient import TestClient

    from examples.fastapi_app import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        check("health endpoint", health["status"] == "ok" and health["indexed"] >= 3)

        response = client.post("/search", json={"query": "payment outage escalation"})
        check("search returns 200", response.status_code == 200)
        data = response.json()
        check(
            "search returns chunks + stats",
            len(data["chunks"]) >= 1 and data["stats"]["kept"] >= 1,
        )
        check(
            "chunk payload complete",
            {"id", "text", "relevance_score", "fusion_score", "streams", "metadata"}
            <= set(data["chunks"][0]),
        )

        blank = client.post("/search", json={"query": ""})
        check("blank query rejected 422", blank.status_code == 422)

        bad = client.post("/search", json={"query": "x", "relevance_threshold": 1.5})
        check("threshold out of range rejected 422", bad.status_code == 422)

        package = client.post("/prompt", json={"query": "payment outage escalation"})
        check(
            "prompt endpoint grounds the system message",
            package.status_code == 200 and "GROUNDING CONTRACT" in package.json()["system"],
        )


# --------------------------------------------------------------------------- #
# .env-driven defaults in app.py                                              #
# --------------------------------------------------------------------------- #


def test_app_env_defaults() -> None:
    print("[app env defaults]")
    from app import parse_args

    keys = ("RAGRAM_TOP_N", "RAGRAM_THRESHOLD", "RAGRAM_STORAGE")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["RAGRAM_TOP_N"] = "2"
        os.environ["RAGRAM_THRESHOLD"] = "0.5"
        os.environ["RAGRAM_STORAGE"] = "postgres"
        args = parse_args([])
        check("top_n from env", args.top_n == 2)
        check("threshold from env", args.threshold == 0.5)
        check("storage backend from env", args.storage == "postgres")
        check("flag still overrides env", parse_args(["--top-n", "3"]).top_n == 3)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    # Lazy-import checks MUST run before any framework module is loaded.
    test_lazy_imports()
    test_normalize_messages()
    test_ollama_generation()
    engine = _make_engine()
    test_answer_pipeline(engine)
    test_openai_generator()
    test_anthropic_generator()
    test_langchain_retriever(engine)
    test_llamaindex_retriever(engine)
    test_fastapi_example()
    test_app_env_defaults()
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
