"""
AgentState definition for Researcher agent.

This is a local copy to ensure the agent is fully self-contained
and can run independently without dependencies on the main repository.
"""

import operator
from typing import Annotated, Sequence, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    context: Dict[str, Any]
    tool_used: bool
