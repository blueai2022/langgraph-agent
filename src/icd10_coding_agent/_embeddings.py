"""Embedding functions for ICD-10 coding."""

from __future__ import annotations

from typing import Protocol, Sequence

from langchain_openai import OpenAIEmbeddings


class Embedder(Protocol):
    """Embeddings interface"""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingBackend:
    """Embedding backend for ICD-10 coding"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key or "sk-",
            check_embedding_ctx_length=False, # to avoid tiktoken requirement
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed_documents(list(texts))
