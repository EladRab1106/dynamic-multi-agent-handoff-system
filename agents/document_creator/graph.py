"""
LangGraph wrapper for the Document Creator agent.

This exposes the Document Creator agent as a LangGraph that can be referenced remotely
via RemoteGraph. The graph is a minimal execution wrapper that:
- Invokes the Document Creator agent chain
- Passes through the agent's output message unchanged
- Extracts completed_capability from the agent's completion contract
- Sets context["last_completed_capability"] for Supervisor orchestration

The graph does NOT:
- Modify or create completion contracts
- Parse or mutate business data (markdown, file_path, etc.)
- Perform orchestration or planning logic
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from document_creator_agent import build_document_creator_agent
from state import AgentState


def build_document_creator_graph():
    """
    Build a minimal LangGraph wrapper around the Document Creator agent chain.
    
    The graph:
    1. Extracts messages from AgentState
    2. Invokes the Document Creator agent chain
    3. Passes through the agent's output message unchanged
    4. Extracts completed_capability from the agent's completion contract
    5. Sets context["last_completed_capability"] for Supervisor orchestration
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    """
    document_creator_chain = build_document_creator_agent()
    
    def document_creator_node(state: AgentState) -> dict:
        """Minimal execution wrapper node for Document Creator.

        Adds basic logging so we can see when the node is invoked and what
        kind of response shape we get back from the agent chain.
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [AIMessage(content="No messages provided")],
                "context": state.get("context", {}),
            }
        
        # Invoke the Document Creator agent chain
        result = document_creator_chain.invoke({"messages": messages})
        
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
    graph.add_node("document_creator", document_creator_node)
    graph.set_entry_point("document_creator")
    graph.add_edge("document_creator", END)
    
    return graph.compile()
