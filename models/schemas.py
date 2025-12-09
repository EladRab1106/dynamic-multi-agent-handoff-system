from typing import List, Literal, TypedDict
from pydantic import BaseModel

PlanStep = Literal["Researcher", "DocumentCreator", "Gmail", "DirectAnswer"]

class Plan(TypedDict):
    steps: List[PlanStep]

class GmailInput(BaseModel):
    action: Literal["search", "send"]
    query: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""
