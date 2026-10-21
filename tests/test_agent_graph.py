import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from icd10_coding_agent.agent_graph import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYSTEM_PROMPT,
    AgentError,
    CaseInput,
    build_coding_graph,
    decide,
)
from icd10_coding_agent.state import FinalAnswer


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        idx = self.calls
        self.calls += 1
        return self.responses[idx]


@tool
def search(query: str) -> str:
    """Search the semantic index for candidate ICD-10 codes."""
    return '[{"code": "E11.42", "display_text": "diabetic polyneuropathy", "score": 0.9}]'


def _answer(code, reason):
    return AIMessage(content=json.dumps({"code": code, "reason": reason}))


def _tool_call(query):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search", "args": {"query": query}, "id": "c1", "type": "tool_call"}
        ],
    )


def test_decide_without_lookup():
    llm = MockLLM([_answer("I10", "matches")])
    graph = build_coding_graph(llm, [search])
    answer = decide(graph, CaseInput(evidence="htn", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert answer.reason == "matches"
    assert llm.calls == 1


def test_decide_after_tool_call():
    llm = MockLLM([_tool_call("diabetic nerve damage"), _answer("E11.42", "found")])
    graph = build_coding_graph(llm, [search])
    answer = decide(
        graph, CaseInput(evidence="DM2 with neuropathy", upstream_candidate_code="E11.9")
    )
    assert answer.code == "E11.42"
    assert llm.calls == 2


def test_decide_never_converging():
    llm = MockLLM([_tool_call("x")] * 100)
    graph = build_coding_graph(llm, [search])
    with pytest.raises(AgentError):
        decide(graph, CaseInput(evidence="...", upstream_candidate_code="I10"), max_steps=5)


def test_decide_malformed_then_valid():
    llm = MockLLM([AIMessage(content="not json"), _answer("I10", "ok")])
    graph = build_coding_graph(llm, [search])
    answer = decide(graph, CaseInput(evidence="...", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert llm.calls == 2


def test_decide_empty_code_then_valid():
    llm = MockLLM([_answer("", ""), _answer("I10", "ok")])
    graph = build_coding_graph(llm, [search])
    answer = decide(graph, CaseInput(evidence="...", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert llm.calls == 2


def test_decide_empty_code_with_reason_accepted():
    llm = MockLLM([_answer("", "no candidate fits the evidence")])
    graph = build_coding_graph(llm, [search])
    answer = decide(graph, CaseInput(evidence="...", upstream_candidate_code="Z99"))
    assert answer.code == ""
    assert answer.reason == "no candidate fits the evidence"
    assert llm.calls == 1


def test_malformed_output_exhausts_retries():
    llm = MockLLM([AIMessage(content="bad")] * 3)
    graph = build_coding_graph(llm, [search], max_retries=2)
    with pytest.raises(AgentError):
        decide(graph, CaseInput(evidence="...", upstream_candidate_code="I10"))


def test_default_bounds():
    assert DEFAULT_MAX_STEPS == 10
    assert DEFAULT_MAX_RETRIES == 2


def test_system_prompt_contains_selection_rules():
    assert "never invent a code" in DEFAULT_SYSTEM_PROMPT
    assert "prior to verify" in DEFAULT_SYSTEM_PROMPT
    assert "Prefer a diagnosis/condition code" in DEFAULT_SYSTEM_PROMPT
    assert "specificity" in DEFAULT_SYSTEM_PROMPT


def test_final_answer_is_pydantic_model():
    answer = FinalAnswer.model_validate_json('{"code": "I10", "reason": "x"}')
    assert answer.code == "I10"
    assert answer.reason == "x"
