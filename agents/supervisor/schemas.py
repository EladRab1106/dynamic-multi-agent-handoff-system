"""
Schemas for Supervisor agent.

Local copy to ensure the agent is fully self-contained.
"""

from typing import List
from pydantic import BaseModel


class Plan(BaseModel):
    steps: List[str]
