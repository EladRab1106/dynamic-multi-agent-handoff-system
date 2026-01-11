import os
import requests
from datetime import datetime
from typing import List, Dict, Any

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a document creation agent. "
    "Convert the latest JSON research in the conversation into a clean Markdown report. "
    "Return ONLY Markdown with headings and bullet points."
)


def messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to dict format for JSON serialization."""
    result = []
    for msg in messages:
        # Use model_dump() for Pydantic v2 or dict() for v1
        if hasattr(msg, 'model_dump'):
            msg_dict = msg.model_dump()
        elif hasattr(msg, 'dict'):
            msg_dict = msg.dict()
        else:
            # Fallback: manual conversion
            msg_dict = {
                "type": msg.__class__.__name__.replace("Message", "").lower(),
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            }
        result.append(msg_dict)
    return result


def message_from_dict(msg_dict: Dict[str, Any]) -> BaseMessage:
    """Convert a message dict back to a LangChain message object."""
    msg_type = msg_dict.get("type", "").lower()
    content = msg_dict.get("content", "")
    
    # Map message types
    if msg_type in ["human", "user"]:
        return HumanMessage(content=content)
    elif msg_type in ["ai", "assistant"]:
        return AIMessage(content=content)
    else:
        # Default to AIMessage
        return AIMessage(content=content)


def build_document_creator_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def document_creator_node(state: AgentState):
    """
    Document Creator node that calls the Document Creator Agent API service.
    
    This adapter makes HTTP requests to the Document Creator service running at
    http://localhost:8002 instead of running the agent locally.
    """
    # Get the API service URL from environment or use default
    api_base_url = os.getenv("DOCUMENT_CREATOR_SERVICE_URL", "http://localhost:8002")
    
    # Extract messages from the state
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in state")
    
    try:
        # Prepare messages for the API call
        # Convert LangChain messages to dict format using LangChain's utilities
        messages_dict = messages_to_dict(list(messages))
        
        # Call the Document Creator Agent API service
        # Using /agent/invoke to get the full LangChain message format
        response = requests.post(
            f"{api_base_url}/agent/invoke",
            json={
                "input": {
                    "messages": messages_dict
                }
            },
            timeout=60  # 60 second timeout (document creation can take time)
        )
        response.raise_for_status()
        
        # Extract the result from the response
        result_data = response.json()
        
        # The LangServe response structure: {"output": <chain_output>}
        # For our agent chain, output is a message object (dict or message)
        output = result_data.get("output")
        
        # Deserialize the message from the API response
        # LangServe returns messages in dict format that can be deserialized
        if isinstance(output, dict):
            # Try to deserialize as a message
            try:
                # If it's a message dict, convert it
                if "type" in output or "content" in output:
                    result = message_from_dict(output)
                else:
                    # Fallback: create AIMessage from content
                    result = AIMessage(content=str(output.get("content", output)))
            except Exception:
                # If deserialization fails, create AIMessage from content
                content = output.get("content", str(output))
                result = AIMessage(content=content)
        elif isinstance(output, str):
            result = AIMessage(content=output)
        else:
            # Fallback: convert to string
            result = AIMessage(content=str(output))
        
        # Extract markdown content
        md_content = result.content if hasattr(result, "content") else str(result)
        
        # Save the markdown content to a file (same as original behavior)
        os.makedirs("outputs", exist_ok=True)
        file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        ctx = dict(state.get("context", {}))
        ctx["file_path"] = file_path
        ctx["last_completed_capability"] = "create_document"
        
        return {
            "messages": [AIMessage(content=f"REPORT_CREATED: {file_path}")],
            "context": ctx,
        }
        
    except requests.exceptions.RequestException as e:
        # Fallback to local execution if API is unavailable
        # This provides resilience in case the service is down
        import warnings
        warnings.warn(
            f"Document Creator API service unavailable ({api_base_url}): {e}. "
            "Falling back to local execution."
        )
        
        # Fallback to local execution
        os.makedirs("outputs", exist_ok=True)
        chain = build_document_creator_agent()
        md_msg = chain.invoke({"messages": state["messages"]})
        
        md_content = md_msg.content if hasattr(md_msg, "content") else str(md_msg)
        
        file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        ctx = dict(state.get("context", {}))
        ctx["file_path"] = file_path
        ctx["last_completed_capability"] = "create_document"
        
        return {
            "messages": [AIMessage(content=f"REPORT_CREATED: {file_path}")],
            "context": ctx,
        }


register_agent(
    AgentSpec(
        name="DocumentCreator",
        capabilities=["create_document", "summarize", "write_document"],
        build_chain=build_document_creator_agent,
    )
)
