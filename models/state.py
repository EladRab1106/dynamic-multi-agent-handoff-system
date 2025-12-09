import operator
from typing import Annotated, Sequence, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    context: Dict[str, Any]
