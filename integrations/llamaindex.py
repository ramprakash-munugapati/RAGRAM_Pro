"""LlamaIndex integration: expose RAGRAM_Pro as a ``BaseRetriever``.

Drop-in usage with query engines and LlamaIndex RAG pipelines::

    from integrations.llamaindex import RagramLlamaIndexRetriever

    retriever = RagramLlamaIndexRetriever(engine, top_n=5, relevance_threshold=0.4)
    nodes = retriever.retrieve("what is the escalation path?")   # list[NodeWithScore]
    query_engine = retriever.as_query_engine(llm=llm)            # full RAG in one line

Each chunk becomes a ``TextNode`` scored by the cross-encoder relevance, so
LlamaIndex similarity thresholds and response synthesizers behave normally.
The sync entry point bridges to RAGRAM's async pipeline safely even inside a
running event loop (see ``integrations/compat.py``).
"""

from __future__ import annotations

try:
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
except ImportError as exc:
    raise ImportError(
        "llama-index-core is not installed. Install it with: "
        "pip install 'ragram-pro[llamaindex]' (or: pip install llama-index-core)"
    ) from exc

from integrations.compat import run_sync
from engine import RagramFusionEngine

__all__ = ["RagramLlamaIndexRetriever"]


class RagramLlamaIndexRetriever(BaseRetriever):
    """LlamaIndex retriever over the RAGRAM hybrid + rerank pipeline."""

    def __init__(
        self,
        engine: RagramFusionEngine,
        *,
        top_n: int | None = None,
        relevance_threshold: float | None = None,
        **kwargs: object,
    ) -> None:
        self._engine = engine
        self._top_n = top_n
        self._relevance_threshold = relevance_threshold
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = run_sync(
            self._engine.search(
                query_bundle.query_str,
                top_n=self._top_n,
                relevance_threshold=self._relevance_threshold,
            )
        )
        nodes: list[NodeWithScore] = []
        for item in result.chunks:
            metadata = dict(item.chunk.metadata)
            metadata.update(
                {
                    "chunk_id": item.chunk.id,
                    "relevance_score": item.relevance_score,
                    "fusion_score": item.fusion_score,
                    "streams": list(item.streams),
                }
            )
            node = TextNode(
                text=item.chunk.text,
                id_=item.chunk.id,
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=item.relevance_score))
        return nodes
