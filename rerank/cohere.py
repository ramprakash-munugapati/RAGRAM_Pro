"""Managed Cohere reranking API backend (``POST /v2/rerank``).

Scores arrive pre-calibrated in ``[0, 1]``, so they drop straight into the
pipeline's ``relevance_threshold``. Uses ``httpx`` lazily; candidates missing
from the API response receive score 0.0 and are removed by threshold filtering.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from models import FusedChunk, RerankedChunk
from rerank.base import Reranker, RerankerError

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("ragram.rerank.cohere")

__all__ = ["CohereReranker"]


class CohereReranker(Reranker):
    """Cloud reranker using Cohere's ``rerank`` endpoint.

    Args:
        api_key: Explicit API key; falls back to the ``COHERE_API_KEY`` env var.
        model: Cohere rerank model (e.g. ``rerank-english-v3.0``, ``rerank-v3.5``).
        base_url: Override for proxy/self-hosted gateways.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
        base_url: str = "https://api.cohere.com",
        api_key_env: str = "COHERE_API_KEY",
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RerankerError(
                f"Cohere API key missing: pass api_key=... or set ${api_key_env}"
            )
        try:
            import httpx
        except ImportError as exc:
            raise RerankerError(
                "httpx is not installed. Install with: pip install 'ragram-pro[cohere]'"
            ) from exc

        self._api_key = key
        self._model = model
        self._timeout = timeout
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[FusedChunk],
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        documents = [candidate.chunk.text for candidate in candidates]
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            # Return every score so the pipeline's relevance_threshold applies
            # uniformly; the engine truncates to top_n afterwards.
            "top_n": len(documents),
        }
        try:
            response = await self._client.post("/v2/rerank", json=payload)
        except Exception as exc:
            raise RerankerError(f"Cohere rerank request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = _error_detail(response)
            raise RerankerError(f"Cohere rerank failed ({response.status_code}): {detail}")

        body = response.json()
        scores_by_index: dict[int, float] = {}
        for item in body.get("results", []):
            try:
                scores_by_index[int(item["index"])] = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed rerank result: %r", item)

        scored = [
            self._pack(candidate, scores_by_index.get(index, 0.0))
            for index, candidate in enumerate(candidates)
        ]
        return self._sort(scored)

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_detail(response: "httpx.Response") -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or body)
    except ValueError:
        pass
    return response.text[:300]
