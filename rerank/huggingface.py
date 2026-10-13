"""Local HuggingFace cross-encoder reranker (default: ``BAAI/bge-reranker-base``).

The cross-encoder reads ``[query, passage]`` jointly and emits a raw logit.
We apply a sigmoid so the output lands in ``[0, 1]`` and is directly comparable
with the pipeline's ``relevance_threshold`` (logit 0 ~ score 0.5 ~ decision
boundary of the model). Inference runs in a worker thread to keep the event
loop responsive.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

from models import FusedChunk, RerankedChunk
from rerank.base import Reranker, RerankerError

__all__ = ["CrossEncoderReranker"]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderReranker(Reranker):
    """On-premise reranker backed by ``sentence_transformers.CrossEncoder``.

    Example::

        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker("mixedbread-ai/mxbai-rerank-large-v1")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str | None = None,
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is not installed. Install it with: "
                "pip install 'ragram-pro[local-rerank]' -- or use CohereReranker / "
                "TermOverlapReranker instead."
            ) from exc
        init_kwargs: dict[str, object] = {}
        if local_files_only:
            init_kwargs["local_files_only"] = True
        try:
            self._model = CrossEncoder(model_name, device=device, **init_kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            raise RerankerError(f"could not load cross-encoder {model_name!r}: {exc}") from exc
        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self,
        query: str,
        candidates: Sequence[FusedChunk],
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        texts = [candidate.chunk.text for candidate in candidates]
        logits = await asyncio.to_thread(self._predict, query, texts)
        scored = [
            self._pack(candidate, _sigmoid(float(logit)))
            for candidate, logit in zip(candidates, logits, strict=True)
        ]
        return self._sort(scored)

    def _predict(self, query: str, texts: Sequence[str]) -> list[float]:
        try:
            pairs = [(query, text) for text in texts]
            raw = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerError(f"cross-encoder inference failed: {exc}") from exc
        # Single-label rerankers return shape (n,); flatten defensively for (n, 1).
        if hasattr(raw, "reshape"):
            raw = raw.reshape(-1)
        return [float(value) for value in raw]
