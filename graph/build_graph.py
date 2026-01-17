# NOTE:
# Validation, enforcement, and correctness checks are intentionally disabled.
# Agents are currently trusted as authoritative.
# A dedicated Validation Agent will be introduced later to verify outputs.

import os
import logging
from typing import Dict
from langgraph.graph import StateGraph, END
from langgraph.pregel.remote import RemoteGraph
from models.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(capability_index: Dict[str, str]):
    """
    Build the Supervisor graph with dynamically discovered agent capabilities.
    
    This function:
    1. Creates RemoteGraph references to all agent services
    2. Maps capabilities to agent names for routing
    3. Builds and returns the compiled graph
    
    Args:
        capability_index: Dict mapping capability strings to agent names
            Example: {"research": "Researcher", "gmail": "Gmail", ...}
    
    Agents are called directly via RemoteGraph - no wrappers, no validation.
    Each agent is the sole authority over when its task is completed.
    
    The orchestrator maps capability strings (from Supervisor) to agent names
    for routing. The Supervisor never sees agent names or URLs.
    """
    graph = StateGraph(AgentState)

    # Create RemoteGraph references to all agents
    # RemoteGraph uses blocking execution when invoked as a node in a graph
    # It waits for the remote agent to complete before returning
    researcher_service_url = os.getenv("RESEARCHER_SERVICE_URL", "http://localhost:8001")
    researcher_remote = RemoteGraph("researcher", url=researcher_service_url)

    document_creator_service_url = os.getenv("DOCUMENT_CREATOR_SERVICE_URL", "http://localhost:8002")
    document_creator_remote = RemoteGraph("document_creator", url=document_creator_service_url)

    gmail_service_url = os.getenv("GMAIL_SERVICE_URL", "http://localhost:8000")
    gmail_remote = RemoteGraph("gmail", url=gmail_service_url)

    supervisor_service_url = os.getenv("SUPERVISOR_SERVICE_URL", "http://localhost:8004")
    supervisor_remote = RemoteGraph("supervisor", url=supervisor_service_url)

    # Build capability → agent_name mapping for routing
    # This mapping is used by the conditional edge to route capability strings to agent nodes
    capability_to_agent: Dict[str, str] = {}
    for capability, agent_name in capability_index.items():
        capability_to_agent[capability] = agent_name

    # Add agents directly - no wrappers, no intermediate logic
    # Supervisor is now a remote service like other agents
    graph.add_node("Supervisor", supervisor_remote)
    graph.add_node("Researcher", researcher_remote)
    graph.add_node("DocumentCreator", document_creator_remote)
    graph.add_node("Gmail", gmail_remote)

    # All agents return directly to Supervisor after completing their work
    graph.add_edge("Researcher", "Supervisor")
    graph.add_edge("DocumentCreator", "Supervisor")
    graph.add_edge("Gmail", "Supervisor")

    def route_from_supervisor(state: AgentState) -> str:
        """
        Route based on Supervisor's output.
        
        Supervisor returns capability strings in state["next"].
        This function maps capability → agent_name for routing.
        """
        next_value = state.get("next", "")
        
        # FINISH is a special routing value, not a capability
        if next_value == "FINISH":
            return "FINISH"
        
        # Map capability string to agent name
        if next_value in capability_to_agent:
            agent_name = capability_to_agent[next_value]
            logger.debug(f"Routing capability '{next_value}' → agent '{agent_name}'")
            return agent_name
        
        # If capability not found, log error and finish
        logger.error(
            f"Unknown capability '{next_value}' from Supervisor. "
            f"Available capabilities: {sorted(capability_to_agent.keys())}. "
            f"Finishing workflow."
        )
        return "FINISH"

    graph.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "Researcher": "Researcher",
            "DocumentCreator": "DocumentCreator",
            "Gmail": "Gmail",
            "FINISH": END,
        }
    )

    graph.set_entry_point("Supervisor")

    logger.info(f"Graph built with capability routing: {len(capability_to_agent)} capabilities mapped to agents")
    return graph.compile()
