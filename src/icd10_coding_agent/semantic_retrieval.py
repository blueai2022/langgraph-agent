"""LangChain-native retrieval over a persisted sqlite-vec index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import field_validator

from ._embeddings import Embedder
from .semantic_indexing import load_index


class RetrievalError(Exception):
    """Raised when a retrieval request is invalid or cannot be served."""


class SQLiteVecRetriever(BaseRetriever):
    """Embeds a query and returns top-k `Document`s ranked by cosine similarity."""

    embeddings: Any  # Embedder (Any to avoid Pydantic Protocol validation)
    index_path: str | Path
    k: int = 5

    @field_validator("k")
    @classmethod
    def _k_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise RetrievalError("k must be positive")
        return value


    def _get_relevant_documents( # abstractmethod at the BaseRetriever level
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        index = load_index(self.index_path)
        try:
            if len(index) == 0:
                raise RetrievalError("the index has no entries")
            if self.embeddings.model != index.model:
                raise RetrievalError(
                    f"query embedding model {self.embeddings.model!r} does not match "
                    f"index model {index.model!r}"
                )
            vectors = self.embeddings.embed([query])
            if len(vectors) != 1:
                raise RetrievalError(
                    f"embedding backend returned {len(vectors)} vectors for one query"
                )
            return [
                Document(
                    page_content=entry.display_text,
                    metadata={
                        "code": entry.code,
                        "cross_references": list(entry.cross_references),
                        "score": round(score, 4),
                    },
                )
                for entry, score in index.search(vectors[0], self.k)
            ]
        finally:
            index.close()
