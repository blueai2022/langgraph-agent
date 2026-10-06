from icd10_coding_agent._embeddings import EmbeddingBackend


def test_embedding_backend_config(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, model=None, base_url=None, api_key=None, **kwargs):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["api_key"] = api_key

        def embed_documents(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setattr("icd10_coding_agent._embeddings.OpenAIEmbeddings", FakeEmbeddings)

    backend = EmbeddingBackend(
        model="mxbai-embed-large",
        base_url="http://localhost:11434/v1",
        api_key="sk-test",
    )
    vectors = backend.embed(["a", "b"])

    assert captured["model"] == "mxbai-embed-large"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "sk-test"
    assert vectors == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_embedding_backend_placeholder_key_when_none(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["api_key"] = kwargs.get("api_key")

        def embed_documents(self, texts):
            return [[0.0] for _ in texts]

    monkeypatch.setattr("icd10_coding_agent._embeddings.OpenAIEmbeddings", FakeEmbeddings)

    EmbeddingBackend(model="m").embed(["a"])
    assert captured["api_key"] == "sk-"


def test_embedding_backend_empty_input():
    backend = EmbeddingBackend(model="m")
    assert backend.embed([]) == []
