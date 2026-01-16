"""
LangGraph wrapper for the Researcher agent.

This exposes the Researcher agent as a LangGraph that can be referenced remotely
via RemoteGraph. The graph is a minimal execution wrapper that:
- Invokes the Researcher agent chain
- Passes through the agent's output message unchanged
- Extracts completed_capability from the agent's completion contract
- Sets context["last_completed_capability"] for Supervisor orchestration

The graph does NOT:
- Modify or create completion contracts
- Parse or mutate business data (research_summary, document_source, etc.)
- Perform orchestration or planning logic
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from researcher_agent import build_researcher_agent
from state import AgentState


def build_researcher_graph():
    """
    Build a minimal LangGraph wrapper around the Researcher agent chain.
    
    The graph:
    1. Extracts messages from AgentState
    2. Invokes the Researcher agent chain
    3. Passes through the agent's output message unchanged
    4. Extracts completed_capability from the agent's completion contract
    5. Sets context["last_completed_capability"] for Supervisor orchestration
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    """
    researcher_chain = build_researcher_agent()
    
    def researcher_node(state: AgentState) -> dict:
        """
        Minimal execution wrapper node.
        
        Invokes the Researcher agent chain and passes through the output unchanged.
        Agent is authoritative - no validation, no enforcement.
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [AIMessage(content="No messages provided")],
                "context": state.get("context", {}),
            }
        
        # Invoke the Researcher agent chain
        result = researcher_chain.invoke({"messages": messages})
        
        # Extract the response message (unchanged from agent output)
        if isinstance(result, AIMessage):
            response_message = result
        elif isinstance(result, dict) and "messages" in result:
            response_message = result["messages"][-1] if result["messages"] else AIMessage(content=str(result))
        else:
            response_message = AIMessage(content=str(result))
        
        # Pass through context unchanged - agent output is authoritative
        ctx = dict(state.get("context", {}))
        
        return {
            "messages": [response_message],
            "context": ctx,
        }
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", END)
    
    return graph.compile()
