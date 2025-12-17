from typing import List, Literal, Optional
from pydantic import BaseModel


class Plan(BaseModel):
    steps: List[str]


class GmailInput(BaseModel):
    action: Literal["search", "send"]

    query: str = ""

    recipient: str = ""
    subject: str = ""
    body: str = ""

    attachments: Optional[List[str]] = None
