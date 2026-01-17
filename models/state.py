import operator
from typing import Annotated, Sequence, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    context: Dict[str, Any]
