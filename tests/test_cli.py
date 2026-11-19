from pathlib import Path

from typer.testing import CliRunner

from icd10_coding_agent.cli import Settings, app, resolve_settings
from icd10_coding_agent.icd10_corpus import Corpus
from icd10_coding_agent.state import FinalAnswer

runner = CliRunner()


def test_settings_builtin_defaults():
    settings = Settings()
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.model == "llama3.1:8b"
    assert settings.rag_embed_base_url == "http://localhost:11434/v1"
    assert settings.rag_embed_model == "mxbai-embed-large"
    assert settings.index_path == Path("index.db")
    assert settings.corpus_path == Path("corpus.json")


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_RAG_EMBED_MODEL", "custom-model")
    monkeypatch.setenv("ICD10_AGENT_INDEX_PATH", "/tmp/foo.db")
    settings = Settings()
    assert settings.rag_embed_model == "custom-model"
    assert settings.index_path == Path("/tmp/foo.db")


def test_settings_llm_env_names(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_BASE_URL", "http://h:1234/v1")
    monkeypatch.setenv("ICD10_AGENT_MODEL", "llama3.2")
    settings = Settings()
    assert settings.base_url == "http://h:1234/v1"
    assert settings.model == "llama3.2"


def test_resolve_settings_cli_over_env(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_RAG_EMBED_MODEL", "env-model")
    assert resolve_settings(rag_embed_model="cli-model").rag_embed_model == "cli-model"


def test_resolve_settings_env_over_default(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_MODEL", "env-llm")
    assert resolve_settings().model == "env-llm"


def test_commands_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index-build" in result.output
    assert "run" in result.output


def test_run_requires_evidence():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_index_build_wires(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]", encoding="utf-8")
    out = tmp_path / "out.db"

    monkeypatch.setattr("icd10_coding_agent.cli.load_corpus", lambda p: Corpus([]))
    monkeypatch.setattr("icd10_coding_agent.cli.load_aliases", lambda p: [])
    monkeypatch.setattr("icd10_coding_agent.cli.validate_aliases", lambda a, c: None)
    captured = {}

    def fake_build(corpus_obj, embeddings, index_path, aliases=()):
        captured["index"] = index_path
        captured["model"] = embeddings.model
        captured["base_url"] = embeddings.base_url

    monkeypatch.setattr("icd10_coding_agent.cli.build_index", fake_build)

    result = runner.invoke(
        app,
        [
            "index-build",
            "--corpus",
            str(corpus),
            "--index",
            str(out),
            "--rag-embed-model",
            "mymodel",
            "--rag-embed-base-url",
            "http://h",
        ],
    )
    assert result.exit_code == 0
    assert captured["index"] == out
    assert captured["model"] == "mymodel"
    assert captured["base_url"] == "http://h"


def _patch_run_deps(monkeypatch, result):
    monkeypatch.setattr(
        "icd10_coding_agent.cli.ChatBackend",
        lambda **kw: type("B", (), {"llm": object()})(),
    )
    monkeypatch.setattr("icd10_coding_agent.cli.EmbeddingBackend", lambda **kw: object())
    monkeypatch.setattr("icd10_coding_agent.cli.SQLiteVecRetriever", lambda **kw: object())
    monkeypatch.setattr("icd10_coding_agent.cli.build_search_tool", lambda r: object())
    monkeypatch.setattr("icd10_coding_agent.cli.build_coding_graph", lambda llm, tool: object())
    monkeypatch.setattr(
        "icd10_coding_agent.cli.invoke_graph",
        lambda graph, case, system_prompt=None, max_steps=10: result,
    )


def test_run_verbose_prints_transcript(monkeypatch):
    from langchain_core.messages import AIMessage, SystemMessage

    result = {
        "messages": [
            SystemMessage(content="SYS"),
            AIMessage(content='{"code": "I10", "reason": "x"}'),
        ],
        "final_answer": FinalAnswer(code="I10", reason="x"),
    }
    _patch_run_deps(monkeypatch, result)

    out = runner.invoke(app, ["run", "the evidence", "I10", "--verbose"])
    assert out.exit_code == 0
    assert "[system] SYS" in out.output
    assert '"code": "I10"' in out.output


def test_run_non_verbose_only_final_decision(monkeypatch):
    from langchain_core.messages import SystemMessage

    result = {
        "messages": [SystemMessage(content="SYS")],
        "final_answer": FinalAnswer(code="I10", reason="x"),
    }
    _patch_run_deps(monkeypatch, result)

    out = runner.invoke(app, ["run", "the evidence", "I10"])
    assert out.exit_code == 0
    assert "[system]" not in out.output
    assert '"code": "I10"' in out.output
