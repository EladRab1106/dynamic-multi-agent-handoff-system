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

from langgraph.graph import StateGraph, END
from supervisor import supervisor_node
from state import AgentState


def build_supervisor_graph():
    """
    Build a minimal LangGraph wrapper around the Supervisor node.
    
    The graph:
    1. Extracts state from AgentState
    2. Invokes the Supervisor node
    3. Passes through the Supervisor's output unchanged
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    """
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", END)
    
    return graph.compile()
