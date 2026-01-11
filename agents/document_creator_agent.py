import os
import requests
from datetime import datetime
from typing import List, Dict, Any

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import build_completion_message, extract_completed_capability

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a document creation agent. "
    "Convert the latest JSON research in the conversation into a clean Markdown report. "
    "Return ONLY Markdown with headings and bullet points. "
    "After completing the document, return JSON: "
    '{{"completed_capability": "create_document", "data": {{"file_path": "outputs/report_xxx.md"}}}}'
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
        
        # Try to parse as completion contract - if so, extract markdown from data
        from agents.utils import parse_completion_message
        contract = parse_completion_message(md_content)
        if contract and "data" in contract and "file_path" in contract["data"]:
            # Agent already returned contract with file_path
            file_path = contract["data"]["file_path"]
            # Try to extract markdown from contract if present
            if "markdown" in contract["data"]:
                md_content = contract["data"]["markdown"]
        else:
            # Extract markdown from content (remove any JSON if present)
            try:
                import json
                parsed = json.loads(md_content)
                if isinstance(parsed, dict) and "data" in parsed and "markdown" in parsed["data"]:
                    md_content = parsed["data"]["markdown"]
            except:
                pass
        
        # Save the markdown content to a file
        os.makedirs("outputs", exist_ok=True)
        file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        ctx = dict(state.get("context", {}))
        ctx["file_path"] = file_path
        
        # Build completion message with contract
        completion_message = build_completion_message("create_document", {"file_path": file_path})
        ctx["last_completed_capability"] = "create_document"
        
        return {
            "messages": [completion_message],
            "context": ctx,
        }
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"doc creator Agent service is unavailable at {api_base_url}. "
            f"Remote execution is required. Original error: {e}"
        )


register_agent(
    AgentSpec(
        name="DocumentCreator",
        capabilities=["create_document", "summarize", "write_document"],
        build_chain=build_document_creator_agent,
    )
)
