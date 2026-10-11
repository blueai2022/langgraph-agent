import pytest

from icd10_coding_agent.code_aliasing import Alias
from icd10_coding_agent.icd10_corpus import Corpus, CorpusEntry
from icd10_coding_agent.semantic_indexing import build_index
from icd10_coding_agent.semantic_retrieval import RetrievalError, SQLiteVecRetriever


class DictEmbedder:
    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[text] for text in texts]


@pytest.fixture
def index_path(tmp_path):
    corpus = Corpus(
        [
            CorpusEntry(code="I10", description="alpha", category="I10", billable=True),
            CorpusEntry(code="E11.9", description="beta", category="E11", billable=True),
        ]
    )
    aliases = [Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")]
    mapping = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "HTN": [0.0, 0.0, 1.0],
    }
    path = tmp_path / "index.db"
    build_index(corpus, DictEmbedder("model-A", mapping), path, aliases=aliases)
    return path


def test_retriever_returns_ordered_documents(index_path):
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-A", {"q": [0.9, 0.1, 0.0]}),
        index_path=str(index_path),
        k=2,
    )
    docs = retriever.invoke("q")
    assert [doc.metadata["code"] for doc in docs] == ["I10", "E11.9"]
    assert docs[0].metadata["score"] > docs[1].metadata["score"]


def test_retriever_k_exceeds_size(index_path):
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-A", {"q": [0.9, 0.1, 0.0]}),
        index_path=str(index_path),
        k=10,
    )
    docs = retriever.invoke("q")
    assert len(docs) == 3


def test_retriever_model_mismatch(index_path):
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-B", {"q": [0.9, 0.1, 0.0]}),
        index_path=str(index_path),
        k=2,
    )
    with pytest.raises(RetrievalError):
        retriever.invoke("q")


def test_retriever_plain_entry_page_content(index_path):
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-A", {"q": [1.0, 0.0, 0.0]}),
        index_path=str(index_path),
        k=1,
    )
    docs = retriever.invoke("q")
    assert docs[0].page_content == "alpha"
    assert docs[0].metadata["code"] == "I10"


def test_retriever_alias_display_text(index_path):
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-A", {"q": [0.0, 0.0, 1.0]}),
        index_path=str(index_path),
        k=1,
    )
    docs = retriever.invoke("q")
    assert docs[0].page_content == "Hypertension - HTN"
    assert docs[0].metadata["code"] == "I10"


def test_retriever_empty_index(tmp_path):
    path = tmp_path / "empty.db"
    build_index(Corpus([]), DictEmbedder("model-A", {}), path)
    retriever = SQLiteVecRetriever(
        embeddings=DictEmbedder("model-A", {"q": [0.0, 0.0, 0.0]}),
        index_path=str(path),
        k=1,
    )
    with pytest.raises(RetrievalError):
        retriever.invoke("q")


def test_retriever_rejects_non_positive_k(index_path):
    with pytest.raises(RetrievalError):
        SQLiteVecRetriever(
            embeddings=DictEmbedder("model-A", {}), index_path=str(index_path), k=0
        )
    with pytest.raises(RetrievalError):
        SQLiteVecRetriever(
            embeddings=DictEmbedder("model-A", {}), index_path=str(index_path), k=-1
        )
