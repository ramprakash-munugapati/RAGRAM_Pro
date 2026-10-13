"""FastAPI microservice exposing RAGRAM_Pro as REST endpoints.

Run (from the repo root):

    pip install fastapi uvicorn          # or: pip install 'ragram-pro[api]'
    uvicorn examples.fastapi_app:app --reload

or directly (host/port come from the environment):

    python examples/fastapi_app.py       # API_HOST / API_PORT from .env

Configuration is externalized into `.env` (copy `.env.example`): the same
RAGRAM_STORAGE / RAGRAM_EMBEDDER / RAGRAM_RERANKER / RAGRAM_FUSION /
RAGRAM_RRF_K / RAGRAM_THRESHOLD / RAGRAM_TOP_N / RAGRAM_CANDIDATE_LIMIT
properties drive this example and `python app.py`, plus API_HOST / API_PORT.

Endpoints:

    GET  /health   -> service status + indexed chunk count
    POST /search   -> hybrid search (fused + reranked chunks + stats)
    POST /prompt   -> grounded generation prompt (system + user) for any LLM API

Example-only code: the RAGRAM_Pro library itself never imports FastAPI.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python examples/fastapi_app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import MOCK_ROWS, build_embedder, build_reranker, build_storage  # noqa: E402
from config import FusionConfig  # noqa: E402
from engine import RagramFusionEngine  # noqa: E402
from envfile import env_float, env_int, env_str, load_env  # noqa: E402

load_env()  # externalized properties: .env / $RAGRAM_ENV_FILE (shell env wins)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language question")
    top_n: int | None = Field(default=None, ge=1, description="Override config.top_n")
    relevance_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override config.relevance_threshold"
    )


class ChunkOut(BaseModel):
    id: str
    text: str
    relevance_score: float
    fusion_score: float
    streams: list[str]
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    chunks: list[ChunkOut]
    stats: dict[str, Any]


class PromptResponse(BaseModel):
    system: str
    user: str
    chunks: list[ChunkOut]
    stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    indexed: int


def _build_config() -> FusionConfig:
    return FusionConfig(
        candidate_limit=env_int("RAGRAM_CANDIDATE_LIMIT", 25),
        fusion_method=env_str("RAGRAM_FUSION", "rrf"),
        rrf_k=env_int("RAGRAM_RRF_K", 60),
        relevance_threshold=env_float("RAGRAM_THRESHOLD", 0.3),
        top_n=env_int("RAGRAM_TOP_N", 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = build_embedder(env_str("RAGRAM_EMBEDDER", "hashing"))
    storage = build_storage(
        env_str("RAGRAM_STORAGE", "memory"), dimensions=embedder.dimensions
    )
    reranker = build_reranker(env_str("RAGRAM_RERANKER", "noop"))
    engine = RagramFusionEngine(storage, embedder, reranker, _build_config())
    async with engine:
        await engine.index(list(MOCK_ROWS))
        app.state.engine = engine
        app.state.storage = storage
        yield


app = FastAPI(
    title="RAGRAM_Pro demo API",
    description="Hybrid search (dense + sparse) + fusion + reranking as a service.",
    version="8.0.0",
    lifespan=lifespan,
)


def _chunks(result: Any) -> list[ChunkOut]:
    return [
        ChunkOut(
            id=item.chunk.id,
            text=item.chunk.text,
            relevance_score=item.relevance_score,
            fusion_score=item.fusion_score,
            streams=list(item.streams),
            metadata=dict(item.chunk.metadata),
        )
        for item in result.chunks
    ]


async def _search(request: SearchRequest):  # type: ignore[no-untyped-def]
    try:
        return await app.state.engine.search(
            request.query,
            top_n=request.top_n,
            relevance_threshold=request.relevance_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed=await app.state.storage.count())


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    result = await _search(request)
    return SearchResponse(
        query=result.query,
        chunks=_chunks(result),
        stats=asdict(result.stats),
    )


@app.post("/prompt", response_model=PromptResponse)
async def prompt(request: SearchRequest) -> PromptResponse:
    try:
        package = await app.state.engine.build_prompt(
            request.query,
            top_n=request.top_n,
            relevance_threshold=request.relevance_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PromptResponse(
        system=package.system,
        user=package.user,
        chunks=_chunks(package.result),
        stats=asdict(package.result.stats),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=env_str("API_HOST", "127.0.0.1"),
        port=env_int("API_PORT", 8000),
    )
