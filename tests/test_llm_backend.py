from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from icd10_coding_agent.llm_backend import ChatBackend, build_search_tool


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [
            Document(
                page_content="Type 2 diabetes mellitus with diabetic polyneuropathy",
                metadata={"code": "E11.42", "cross_references": [], "score": 0.82},
            ),
            Document(
                page_content="Essential (primary) hypertension",
                metadata={"code": "I10", "cross_references": [], "score": 0.31},
            ),
        ]


def test_chat_backend_config(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, model=None, base_url=None, api_key=None, **kwargs):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr("icd10_coding_agent.llm_backend.ChatOpenAI", FakeChatOpenAI)

    backend = ChatBackend(
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        api_key="sk-test",
    )
    assert backend.model == "llama3.1:8b"
    assert backend.base_url == "http://localhost:11434/v1"
    assert captured["model"] == "llama3.1:8b"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "sk-test"


def test_chat_backend_placeholder_key_when_none(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["api_key"] = kwargs.get("api_key")

    monkeypatch.setattr("icd10_coding_agent.llm_backend.ChatOpenAI", FakeChatOpenAI)

    ChatBackend(model="m")
    assert captured["api_key"] == "sk-local"


def test_search_tool_name_and_query_arg():
    tool = build_search_tool(FakeRetriever())
    assert tool.name == "search"
    assert "query" in tool.args_schema.model_fields


def test_search_tool_preserves_code_and_score():
    tool = build_search_tool(FakeRetriever())
    result = tool.invoke({"query": "diabetic nerve damage"})
    assert "E11.42" in result
    assert "0.82" in result
    assert "Type 2 diabetes mellitus with diabetic polyneuropathy" in result
