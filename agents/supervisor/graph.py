"""
LangGraph wrapper for the Supervisor agent.

This exposes the Supervisor as a LangGraph that can be referenced remotely
via RemoteGraph. The graph is a minimal execution wrapper that:
- Invokes the Supervisor node function
- Passes through the Supervisor's output unchanged

The graph does NOT:
- Perform orchestration or planning logic
- Dispatch other agents
- Call RemoteGraph
- Loop or iterate
"""

import os
import logging

from langgraph.graph import StateGraph, END
from supervisor import supervisor_node
from state import AgentState

logger = logging.getLogger(__name__)


def build_supervisor_graph():
    """
    Build a minimal LangGraph wrapper around the Supervisor node.
    
    The graph:
    1. Extracts state from AgentState
    2. Invokes the Supervisor node
    3. Passes through the Supervisor's output unchanged
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    
    AZTM login is **optional** here. To enable it inside the Supervisor
    service, set SUPERVISOR_AZTM_LOGIN=true in the environment. By default
    it is disabled to avoid event-loop conflicts when running under
    async servers (e.g. langgraph dev / FastAPI / Cloud Run).
    """
    # enable_aztm = os.environ.get("SUPERVISOR_AZTM_LOGIN", "false").lower() in {"1", "true", "yes"}
    AZTM_DISABLE_AUTO_PATCH=1
    import aztm 

    # Prefer AZTM_USERID if present, otherwise fall back to AZTM_JID from the agent's .env

    user="rabiapi" 
    server_mode="true"  

    aztm.login(userid=user, password="apipass", server_mode=server_mode)

    logger.info("AZTM login successful for '%s' (server_mode=%s)", user, server_mode)

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", END)
    
    return graph.compile()
