from typing import Dict
from agents.spec import AgentSpec

AGENT_REGISTRY: Dict[str, AgentSpec] = {}
CAPABILITY_INDEX: Dict[str, str] = {}


def register_agent(spec: AgentSpec):
    AGENT_REGISTRY[spec.name] = spec
    for cap in spec.capabilities:
        CAPABILITY_INDEX[cap] = spec.name
