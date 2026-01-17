"""
LangGraph wrapper for the Direct Answer agent.

This exposes the Direct Answer agent as a LangGraph that can be referenced remotely
via RemoteGraph. The graph is a minimal execution wrapper that:
- Invokes the Direct Answer agent chain
- Passes through the agent's output message unchanged
- Extracts completed_capability from the agent's completion contract
- Sets context["last_completed_capability"] for Supervisor orchestration

The graph does NOT:
- Modify or create completion contracts
- Parse or interpret answer content
- Perform orchestration or planning logic
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from direct_answer_agent import build_direct_answer_agent
from state import AgentState


def build_direct_answer_graph():
    """
    Build a minimal LangGraph wrapper around the Direct Answer agent chain.
    
    The graph:
    1. Extracts messages from AgentState
    2. Invokes the Direct Answer agent chain
    3. Passes through the agent's output message unchanged
    4. Extracts completed_capability from the agent's completion contract
    5. Sets context["last_completed_capability"] for Supervisor orchestration
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    """
    direct_answer_chain = build_direct_answer_agent()
    
    def direct_answer_node(state: AgentState) -> dict:
        """
        Minimal execution wrapper node.
        
        Invokes the Direct Answer agent chain and propagates the completed capability
        into context for Supervisor orchestration. The agent's output message is
        passed through unchanged.
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [AIMessage(content="No messages provided")],
                "context": state.get("context", {}),
            }
        
        # Invoke the Direct Answer agent chain
        result = direct_answer_chain.invoke({"messages": messages})
        
        # Extract the response message (unchanged from agent output)
        if isinstance(result, AIMessage):
            response_message = result
        elif isinstance(result, dict) and "messages" in result:
            response_message = result["messages"][-1] if result["messages"] else AIMessage(content=str(result))
        else:
            response_message = AIMessage(content=str(result))
        
        # Update context: extract completed_capability for Supervisor orchestration
        ctx = dict(state.get("context", {}))
        content = response_message.content if hasattr(response_message, 'content') else str(response_message)
        
        # Pass through context unchanged - agent output is authoritative
        # No validation, no enforcement
        
        return {
            "messages": [response_message],
            "context": ctx,
        }
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("direct_answer", direct_answer_node)
    graph.set_entry_point("direct_answer")
    graph.add_edge("direct_answer", END)
    
    return graph.compile()
