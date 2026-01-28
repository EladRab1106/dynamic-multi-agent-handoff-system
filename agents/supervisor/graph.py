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
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from supervisor import supervisor_node
from state import AgentState
import aztm

logger = logging.getLogger(__name__)




# Prefer AZTM_USERID if present, otherwise fall back to AZTM_JID from the agent's .env

user="rabiapi@connect.mishtamesh.net"
server_mode="true"  

#aztm.login(user, password="apipass", server_mode=server_mode)
aztm.login(
            userid=user,
            password="apipass",
            host="connect.mishtamesh.net",
            port=443,
            tls_mode="direct",
            validate_cert=False,
            server_mode=server_mode)

logger.info("AZTM login successful for '%s' (server_mode=%s)", "rabiapi@connect.mishtamesh.net", server_mode)

def _safe_plan_len(ctx: Any) -> Optional[int]:
    if not isinstance(ctx, dict):
        return None
    plan = ctx.get("plan")
    return len(plan) if isinstance(plan, list) else None


def _supervisor_node_with_logging(state: AgentState) -> Dict[str, Any]:
    """Invoke supervisor_node with high-signal, low-leak logging."""
    ctx_in = state.get("context") if isinstance(state, dict) else None

    try:
        msgs = state.get("messages") if isinstance(state, dict) else None
        msg_count = len(msgs) if msgs is not None else None
    except Exception:
        msg_count = None

    logger.info(
        "Supervisor node invoked (msg_count=%s, current_step_index=%s, plan_len=%s, next=%s)",
        msg_count,
        (ctx_in or {}).get("current_step_index") if isinstance(ctx_in, dict) else None,
        _safe_plan_len(ctx_in),
        state.get("next") if isinstance(state, dict) else None,
    )

    try:
        out = supervisor_node(state)
    except Exception:
        logger.exception(
            "Supervisor node crashed (current_step_index=%s, plan_len=%s)",
            (ctx_in or {}).get("current_step_index") if isinstance(ctx_in, dict) else None,
            _safe_plan_len(ctx_in),
        )
        raise

    ctx_out = out.get("context") if isinstance(out, dict) else None
    logger.info(
        "Supervisor node completed (next=%s, current_step_index=%s, plan_len=%s)",
        out.get("next") if isinstance(out, dict) else None,
        (ctx_out or {}).get("current_step_index") if isinstance(ctx_out, dict) else None,
        _safe_plan_len(ctx_out),
    )
    logger.debug(
        "Supervisor node output keys=%s",
        sorted(list(out.keys())) if isinstance(out, dict) else type(out).__name__,
    )

    return out


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
    

    logger.info("Building Supervisor LangGraph")

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor_node_with_logging)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", END)

    compiled = graph.compile()
    logger.info("Supervisor LangGraph compiled")
    return compiled
