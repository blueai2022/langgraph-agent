"""A LangGraph ``StateGraph`` that decides a final ICD-10 code for a single case."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ValidationError

from .state import AgentState, FinalAnswer

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_RETRIES = 2


class AgentError(Exception):
    """Raised when the agent cannot produce a final answer."""


class CaseInput(BaseModel):
    """Input for a single coding decision."""

    evidence: str
    upstream_candidate_code: str


DEFAULT_SYSTEM_PROMPT = (
    "You are an ICD-10 medical coding assistant. You decide the final ICD-10 code for a "
    "single case, given quoted clinical evidence and an upstream candidate code.\n"
    "You may call the `search` tool any number of times to retrieve candidate codes and "
    "their descriptions from a semantic index; call it whenever you need more evidence.\n"
    "Selection rules:\n"
    "- Choose a final code only from candidates surfaced by `search`; never invent a code "
    "from memory.\n"
    "- Treat the upstream candidate code as a prior to verify, not a certainty to keep "
    "unconditionally.\n"
    "- Prefer a diagnosis/condition code over a symptom/finding code when both plausibly "
    "fit, except that a complication of a diagnosis is itself a condition.\n"
    "- Match the specificity of the evidence: select a specific complication/qualifier "
    "code when the evidence supports it, and an unspecified code only when it does not.\n"
    "When you are ready to conclude, respond with a single JSON object and nothing else, "
    'of the form {"code": "<ICD-10 code, or empty string if none fits>", "reason": '
    '"<brief explanation>"}.\n'
)


def _parse_answer(content: str | None) -> FinalAnswer | None:
    """Small local models tend to output non-JSON text, so we don't raise errors; just return None."""
    if content is None:
        return None
    try:
        return FinalAnswer.model_validate_json(content)
    except ValidationError:
        return None


def _retry_prompt(answer: FinalAnswer | None) -> str:
    if answer is None:
        return (
            "Your response was not a valid JSON object with 'code' and 'reason' fields. "
            "Retry with a schema-conformant answer."
        )
    return (
        "Your final code was empty. Reconsider the retrieved candidates before concluding: "
        "provide a code, or a reason explaining why none fit."
    )


def build_coding_graph(
    llm: Runnable,
    tools: Sequence[BaseTool],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> CompiledStateGraph:
    """Build and compile the coding agent as a LangGraph `StateGraph`."""
    model = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState) -> dict[str, Any]:
        reply = model.invoke(state["messages"])
        return {"messages": [reply]}

    def route_after_model(state: AgentState) -> Literal["tools", "validate"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "validate"

    def validate(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        content = last.content if isinstance(last, AIMessage) else None
        answer = _parse_answer(content)
        if answer is not None and (answer.code or answer.reason):
            return {"final_answer": answer}
        retries = int(state.get("retries", 0)) + 1
        if retries > max_retries:
            raise AgentError(f"model failed to produce a valid answer after {max_retries} retries")
        return {
            "retries": retries,
            "messages": [HumanMessage(content=_retry_prompt(answer))],
        }

    def route_after_validate(state: AgentState) -> Literal["call_model"] | str:
        if state.get("final_answer") is not None:
            return END
        return "call_model"

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("validate", validate)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model", route_after_model, {"tools": "tools", "validate": "validate"}
    )
    graph.add_edge("tools", "call_model")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"call_model": "call_model", END: END}
    )
    return graph.compile()


def _initial_state(case: CaseInput, system_prompt: str) -> dict[str, Any]:
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=_user_message(case)),
        ],
        "evidence": case.evidence,
        "upstream_candidate_code": case.upstream_candidate_code,
        "retries": 0,
        "final_answer": None,
    }


def _user_message(case: CaseInput) -> str:
    return (
        f"Clinical evidence:\n{case.evidence}\n\n"
        f"Upstream candidate code: {case.upstream_candidate_code}\n\n"
        "Determine the final ICD-10 code for this case."
    )


def invoke_graph(
    graph: CompiledStateGraph,
    case: CaseInput,
    *,
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Invoke the graph and return the final state (messages + final answer)."""
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    try:
        return graph.invoke(
            _initial_state(case, prompt),
            config={"recursion_limit": max_steps},
        )
    except GraphRecursionError as exc:
        raise AgentError(
            f"model reasoning failed to produce an answer within {max_steps} steps"
        ) from exc


def decide(
    graph: CompiledStateGraph,
    case: CaseInput,
    *,
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> FinalAnswer:
    """Run the graph and return the final structured decision."""
    result = invoke_graph(graph, case, system_prompt=system_prompt, max_steps=max_steps)
    answer = result.get("final_answer")
    if answer is None:
        raise AgentError("model failed to produce a final answer")
    return answer
