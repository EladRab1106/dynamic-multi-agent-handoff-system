from typing import Dict
from agents.spec import AgentSpec

AGENT_REGISTRY: Dict[str, AgentSpec] = {}
CAPABILITY_INDEX: Dict[str, str] = {}


def register_agent(spec: AgentSpec):
    """
    Legacy registration function for backward compatibility.
    
    Note: In a fully decoupled system, capabilities are discovered dynamically
    via agent service metadata endpoints. This function is kept for compatibility
    but should not be relied upon for capability discovery.
    """
    AGENT_REGISTRY[spec.name] = spec
    for cap in spec.capabilities:
        CAPABILITY_INDEX[cap] = spec.name


def set_capability_index(index: Dict[str, str]):
    """
    Set the capability index dynamically (used by capability discovery).
    
    This allows the discovery mechanism to populate CAPABILITY_INDEX
    without relying on import-time side effects.
    """
    global CAPABILITY_INDEX
    CAPABILITY_INDEX = index
