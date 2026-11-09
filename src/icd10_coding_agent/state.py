"""Graph state schema and the structured final-answer model."""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class FinalAnswer(BaseModel):
    """Final answer from the coding agent."""

    code: str = ""
    reason: str = ""


class AgentState(BaseModel):
    """Typed graph state: the message transcript plus the case and decision."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    evidence: str
    upstream_candidate_code: str
    retries: int = 0
    final_answer: FinalAnswer | None = None
