"""LangChain integration: expose RAGRAM_Pro as a ``BaseRetriever``.

Drop-in usage with LCEL chains, agents and any LangChain RAG pipeline::

    from integrations.langchain import RagramRetriever

    retriever = RagramRetriever(engine=engine, top_n=5, relevance_threshold=0.4)
    docs = retriever.invoke("what is the escalation path?")      # sync
    docs = await retriever.ainvoke("what is the escalation path?")  # async

``invoke`` runs the hybrid pipeline on a private event loop when the caller
has none, and ``ainvoke`` awaits it directly, so both paths are efficient.
Each chunk becomes a LangChain ``Document`` whose metadata carries the chunk
id, reranker score, fusion score, contributing streams and provenance.
"""

from __future__ import annotations

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict
except ImportError as exc:
    raise ImportError(
        "langchain-core is not installed. Install it with: "
        "pip install 'ragram-pro[langchain]' (or: pip install langchain-core)"
    ) from exc

from integrations.compat import run_sync
from engine import RagramFusionEngine
from models import FusionResult

__all__ = ["RagramRetriever"]


class RagramRetriever(BaseRetriever):
    """LangChain retriever backed by the full RAGRAM hybrid + rerank pipeline."""

    engine: RagramFusionEngine
    top_n: int | None = None
    relevance_threshold: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _search(self, query: str) -> FusionResult:
        return run_sync(
            self.engine.search(
                query,
                top_n=self.top_n,
                relevance_threshold=self.relevance_threshold,
            )
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        del run_manager  # callback plumbing is handled by the base class
        return _to_documents(self._search(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        del run_manager
        result = await self.engine.search(
            query,
            top_n=self.top_n,
            relevance_threshold=self.relevance_threshold,
        )
        return _to_documents(result)


def _to_documents(result: FusionResult) -> list[Document]:
    documents: list[Document] = []
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
        documents.append(
            Document(page_content=item.chunk.text, id=item.chunk.id, metadata=metadata)
        )
    return documents
