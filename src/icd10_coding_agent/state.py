"""Graph state schema and the structured final-answer model."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class FinalAnswer(BaseModel):
    """Final answer from the coding agent."""

    code: str = ""
    reason: str = ""


class AgentState(TypedDict):
    """Typed graph state: the message transcript plus the case and decision."""

    messages: Annotated[list[BaseMessage], add_messages]
    evidence: str
    upstream_candidate_code: str
    retries: int
    final_answer: FinalAnswer | None
