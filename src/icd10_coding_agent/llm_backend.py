"""LangChain chat backend and the retriever-backed ``search`` tool."""

from __future__ import annotations

from langchain_core.prompts import StringPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI


class ChatBackend:
    """A LangChain `ChatOpenAI` chat model, e.g., Llama 3."""

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
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key or "sk-local",
        )


class _DocumentPrompt(StringPromptTemplate):
    """Renders a retrieved `Document` with its code and score kept visible.

    ``create_retriever_tool`` spreads each document's metadata as top-level kwargs,
    so the prompt declares the metadata keys it renders.
    """

    input_variables: list[str] = ["page_content", "code", "score"]

    def format(self, **kwargs) -> str:
        return (
            f"code: {kwargs.get('code', '')}\n"
            f"description: {kwargs.get('page_content', '')}\n"
            f"score: {kwargs.get('score', '')}"
        )


# Each retrieved document is rendered with its code and score kept visible for llm to select.
_DOCUMENT_PROMPT = _DocumentPrompt()


def build_search_tool(retriever: BaseRetriever):
    """Create the ``search`` tool from a retriever, preserving code and score."""
    return create_retriever_tool(
        retriever,
        name="search",
        description="Search the semantic index for candidate ICD-10 codes.",
        document_prompt=_DOCUMENT_PROMPT,
        document_separator="\n\n",
    )
