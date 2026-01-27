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
    """

    # ============================
    # MODE TOGGLE: SUPERVISOR ONLY
    # ============================
    # Default = supervisor-only unless explicitly disabled
    # Set env SUPERVISOR_ONLY_MODE=false to enable all agents
    _supervisor_only_raw = os.getenv("SUPERVISOR_ONLY_MODE")
    if _supervisor_only_raw is None:
        supervisor_only = True
    else:
        supervisor_only = _supervisor_only_raw.lower() not in {"0", "false", "no"}

    if not capability_index and not supervisor_only:
        raise RuntimeError("No capabilities discovered – cannot build graph")

    graph = StateGraph(AgentState)

    # ============================
    # REMOTE AGENTS (KEEP COMMENTED FOR SUPERVISOR-ONLY)
    # Uncomment ONLY when running full multi-agent mode
    # ============================

    # researcher_service_url = os.getenv(
    #     "RESEARCHER_SERVICE_URL",
    #     "https://researcher-agent-724942100863.us-central1.run.app/"
    # )
    # researcher_remote = RemoteGraph("researcher", url=researcher_service_url)

    # document_creator_service_url = os.getenv(
    #     "DOCUMENT_CREATOR_SERVICE_URL",
    #     "http://localhost:8002"
    # )
    # document_creator_remote = RemoteGraph("document_creator", url=document_creator_service_url)

    # gmail_service_url = os.getenv(
    #     "GMAIL_SERVICE_URL",
    #     "http://localhost:8000"
    # )
    # gmail_remote = RemoteGraph("gmail", url=gmail_service_url)

    # ============================
    # SUPERVISOR REMOTEGRAPH (ALWAYS ENABLED)
    # ============================
    supervisor_service_url = os.getenv("SUPERVISOR_SERVICE_URL", "http://localhost:8004")
    supervisor_remote = RemoteGraph("supervisor", url=supervisor_service_url)

    # ============================
    # CAPABILITY → AGENT MAP
    # Used only in multi-agent mode
    # ============================
    capability_to_agent: Dict[str, str] = {}
    for capability, agent_name in capability_index.items():
        capability_to_agent[capability] = agent_name

    # ============================
    # REGISTER GRAPH NODES
    # ============================
    graph.add_node("Supervisor", supervisor_remote)

    # ENABLE THESE ONLY IN MULTI-AGENT MODE
    # graph.add_node("Researcher", researcher_remote)
    # graph.add_node("DocumentCreator", document_creator_remote)
    # graph.add_node("Gmail", gmail_remote)

    # ============================
    # OPTIONAL RETURN PATHS (MULTI-AGENT ONLY)
    # ============================
    # graph.add_edge("Researcher", "Supervisor")
    # graph.add_edge("DocumentCreator", "Supervisor")
    # graph.add_edge("Gmail", "Supervisor")

    def route_from_supervisor(state: AgentState) -> str:
        """
        Supervisor returns a capability string in state["next"].
        """

        next_value = state.get("next", "")

        # FINISH is terminal
        if next_value == "FINISH":
            return "FINISH"

        # ============================
        # SUPERVISOR-ONLY MODE SAFETY
        # Forces finish even if Supervisor tries to call an agent
        # ============================
        if supervisor_only:
            logger.info("Supervisor-only mode active — forcing FINISH")
            return "FINISH"

        # ============================
        # MULTI-AGENT MODE ROUTING
        # ============================
        if next_value in capability_to_agent:
            agent_name = capability_to_agent[next_value]
            logger.debug(f"Routing capability '{next_value}' → agent '{agent_name}'")
            return agent_name

        logger.error(
            f"Unknown capability '{next_value}'. "
            f"Available: {sorted(capability_to_agent.keys())}. "
            f"Finishing workflow."
        )
        return "FINISH"

    # ============================
    # ROUTES CONFIG
    # ============================
    routes = {"FINISH": END}

    # ENABLE ONLY IF MULTI-AGENT MODE
    if not supervisor_only:
        routes.update(
            {
                "Researcher": "Researcher",
                "DocumentCreator": "DocumentCreator",
                "Gmail": "Gmail",
            }
        )

    graph.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        routes,
    )

    graph.set_entry_point("Supervisor")

    # ============================
    # LOGGING MODE
    # ============================
    if supervisor_only:
        logger.info("Graph built in SUPERVISOR-ONLY mode (AZTM test mode)")
    else:
        logger.info(
            f"Graph built in MULTI-AGENT mode: {len(capability_to_agent)} capabilities mapped"
        )

    return graph.compile()
