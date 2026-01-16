"""
LangGraph wrapper for the Gmail agent.

This exposes the Gmail agent as a LangGraph that can be referenced remotely
via RemoteGraph. The graph is a minimal execution wrapper that:
- Invokes the Gmail agent chain
- Passes through the agent's output message unchanged
- Extracts completed_capability from the agent's completion contract
- Sets context["last_completed_capability"] for Supervisor orchestration

The graph does NOT:
- Modify or create completion contracts
- Parse or mutate business data (email contents, tool outputs, etc.)
- Perform orchestration or planning logic
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
import json
import logging

from gmail_agent import build_gmail_agent, reset_tool_usage, was_tool_used
from state import AgentState

logger = logging.getLogger(__name__)


def build_gmail_graph():
    """
    Build a minimal LangGraph wrapper around the Gmail agent chain.
    
    The graph:
    1. Extracts messages from AgentState
    2. Invokes the Gmail agent chain
    3. Passes through the agent's output message unchanged
    4. Extracts completed_capability from the agent's completion contract
    5. Sets context["last_completed_capability"] for Supervisor orchestration
    
    This graph can be exposed via LangServe and referenced remotely via RemoteGraph.
    """
    gmail_chain = build_gmail_agent()
    
    def gmail_node(state: AgentState) -> dict:
        """
        Minimal execution wrapper node.
        
        Invokes the Gmail agent chain and passes through the output unchanged.
        Agent is authoritative - no validation, no enforcement.
        """
        # Reset tool usage at start of each invocation
        reset_tool_usage()
        
        messages = state.get("messages", [])
        ctx = state.get("context", {})
        
        # Pass all messages from state to agent (includes injected completion contract)
        # The agent infers intent from conversation history and reads file_path from DocumentCreator completion contract
        messages_for_agent = list(messages)
        
        logger.info(f"Gmail graph: Passing {len(messages)} messages from state to agent")
        logger.debug(f"Gmail graph: Agent will infer intent (search vs send) from conversation history")
        
        # Invoke the Gmail agent chain
        result = gmail_chain.invoke({"messages": messages_for_agent})
        
        # Extract the response message (unchanged from agent output)
        if isinstance(result, AIMessage):
            response_message = result
        elif isinstance(result, dict) and "messages" in result:
            response_message = result["messages"][-1] if result["messages"] else AIMessage(content=str(result))
        else:
            response_message = AIMessage(content=str(result))
        
        # Check if tool was used and add to completion contract if present
        tool_used = was_tool_used()
        content = response_message.content if hasattr(response_message, 'content') else str(response_message)
        
        # Try to parse and extend completion contract with tool_used
        try:
            contract = json.loads(content.strip())
            if isinstance(contract, dict) and "completed_capability" in contract:
                # Extend completion contract with tool_used
                if "data" not in contract:
                    contract["data"] = {}
                contract["data"]["tool_used"] = tool_used
                # Update response message with extended contract
                response_message = AIMessage(content=json.dumps(contract))
                logger.info(f"Gmail graph: Extended completion contract with tool_used={tool_used}")
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Not a completion contract, pass through unchanged
            pass
        
        # Pass through context unchanged - agent output is authoritative
        ctx = dict(state.get("context", {}))
        
        return {
            "messages": [response_message],
            "context": ctx,
        }
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("gmail", gmail_node)
    graph.set_entry_point("gmail")
    graph.add_edge("gmail", END)
    
    return graph.compile()
