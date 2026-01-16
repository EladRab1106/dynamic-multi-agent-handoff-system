# NOTE:
# Validation, enforcement, and correctness checks are intentionally disabled.
# Agents are currently trusted as authoritative.
# A dedicated Validation Agent will be introduced later to verify outputs.

import os
import logging
from langgraph.graph import StateGraph, END
from langgraph.pregel.remote import RemoteGraph
from agents.supervisor.supervisor import supervisor_node
from models.state import AgentState

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build the Supervisor graph with dynamically discovered agent capabilities.
    
    This function:
    1. Creates RemoteGraph references to all agent services
    2. Builds and returns the compiled graph
    
    Agents are called directly via RemoteGraph - no wrappers, no validation.
    Each agent is the sole authority over when its task is completed.
    
    Note: Capability discovery must be run BEFORE calling this function
    (typically in main.py or system startup) to ensure CAPABILITY_INDEX
    is populated before Supervisor planning begins.
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

    direct_answer_service_url = os.getenv("DIRECT_ANSWER_SERVICE_URL", "http://localhost:8003")
    direct_answer_remote = RemoteGraph("direct_answer", url=direct_answer_service_url)

    # Add agents directly - no wrappers, no intermediate logic
    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Researcher", researcher_remote)
    graph.add_node("DocumentCreator", document_creator_remote)
    graph.add_node("Gmail", gmail_remote)
    graph.add_node("DirectAnswer", direct_answer_remote)

    # All agents return directly to Supervisor after completing their work
    graph.add_edge("Researcher", "Supervisor")
    graph.add_edge("DocumentCreator", "Supervisor")
    graph.add_edge("Gmail", "Supervisor")
    graph.add_edge("DirectAnswer", "Supervisor")

    graph.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "DocumentCreator": "DocumentCreator",
            "Gmail": "Gmail",
            "DirectAnswer": "DirectAnswer",
            "FINISH": END,
        }
    )

    graph.set_entry_point("Supervisor")

    logger.info("Graph built with direct agent calls - agents are authoritative")
    return graph.compile()
