import os
import requests
from typing import Optional, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from tools.gmail_tool import gmail_tool
from config.config import llm
from models.state import AgentState

SYSTEM_PROMPT = (
    "You are the Gmail Agent.\n"
    "You can SEARCH emails or SEND emails.\n"
    "- If sending an email, you may optionally attach files.\n"
    "- Attachments must be real file paths available in the system.\n"
    "- Only attach files if the user explicitly asks for it.\n"
    "- If unsure which file to attach, ask a clarification question.\n"
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
    tool_calls = msg_dict.get("tool_calls", [])
    
    # Map message types
    if msg_type in ["human", "user"]:
        return HumanMessage(content=content)
    elif msg_type in ["ai", "assistant"]:
        # Create AIMessage with tool_calls if present
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        else:
            return AIMessage(content=content)
    else:
        # Default to AIMessage
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        else:
            return AIMessage(content=content)


def build_gmail_agent():
    return create_agent(
        llm=llm,
        tools=[gmail_tool],
        system_prompt=SYSTEM_PROMPT,
    )


def gmail_node(state: AgentState):
    """
    Gmail node that calls the Gmail Agent API service.
    
    This adapter makes HTTP requests to the Gmail service running at
    http://localhost:8000 instead of running the agent locally.
    """
    # Get the API service URL from environment or use default
    api_base_url = os.getenv("GMAIL_SERVICE_URL", "http://localhost:8000")
    
    # Extract the last user message from the state
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in state")
    
    # Get the last message content
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        message_content = last_message.content
    else:
        message_content = str(last_message)
    
    try:
        # Prepare messages for the API call
        # Convert LangChain messages to dict format using LangChain's utilities
        messages_dict = messages_to_dict(list(messages))
        
        # Call the Gmail Agent API service
        # Using /agent/invoke to get the full LangChain message format
        try:
            response = requests.post(
                f"{api_base_url}/agent/invoke",
                json={
                    "input": {
                        "messages": messages_dict
                    }
                },
                timeout=5  # Short timeout to fail fast if service is down
            )
            response.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as api_error:
            # Service is not running, fall through to local execution
            raise api_error
        
        # Extract the result from the response
        result_data = response.json()
        
        # The LangServe response structure: {"output": <chain_output>}
        # For our agent chain, output is a message object (dict or message)
        output = result_data.get("output")
        
        # Deserialize the message from the API response
        # LangServe returns messages in dict format that can be deserialized
        if output is None:
            # Empty response - create a placeholder message
            result = AIMessage(content="Processing request...")
        elif isinstance(output, dict):
            # Try to deserialize as a message
            try:
                # If it's a message dict, convert it
                if "type" in output or "content" in output or "tool_calls" in output:
                    result = message_from_dict(output)
                    # Preserve tool_calls if present - this is critical for ToolNode
                    if "tool_calls" in output:
                        # Ensure tool_calls are properly set
                        tool_calls = output["tool_calls"]
                        if tool_calls:
                            # If result already has tool_calls, merge them
                            if hasattr(result, 'tool_calls') and result.tool_calls:
                                # Already set by message_from_dict
                                pass
                            else:
                                # Set tool_calls directly
                                result.tool_calls = tool_calls
                else:
                    # Fallback: create AIMessage from content
                    content = output.get("content", str(output))
                    if not content:
                        content = "Processing request..."
                    result = AIMessage(content=content)
            except Exception as e:
                # If deserialization fails, create AIMessage from content
                content = output.get("content", str(output))
                tool_calls = output.get("tool_calls", [])
                if not content and not tool_calls:
                    content = f"Processing request... (Error: {str(e)})"
                if tool_calls:
                    result = AIMessage(content=content or "", tool_calls=tool_calls)
                else:
                    result = AIMessage(content=content)
        elif isinstance(output, str):
            if not output:
                output = "Processing request..."
            result = AIMessage(content=output)
        else:
            # Fallback: convert to string
            content = str(output) if output else "Processing request..."
            result = AIMessage(content=content)
        
        # Check if tool calls were made (for context tracking)
        ctx = dict(state.get("context", {}))
        
        # Don't set last_completed_capability here if there are tool calls
        # The Supervisor will check ToolMessage results after GmailTools executes
        # Only set it if there are no tool calls (direct response)
        if not (hasattr(result, 'tool_calls') and result.tool_calls):
            # No tool calls - infer from response content
            content_lower = (result.content if hasattr(result, 'content') else str(result)).lower()
            if "email_sent" in content_lower or "sent successfully" in content_lower:
                ctx["last_completed_capability"] = "send_email"
            elif "from:" in content_lower or "subject:" in content_lower:
                ctx["last_completed_capability"] = "search_email"

        return {
            "messages": [result],
            "context": ctx,
        }

        
    except requests.exceptions.RequestException as e:
        # Fallback to local execution if API is unavailable
        # This provides resilience in case the service is down
        import warnings
        warnings.warn(
            f"Gmail API service unavailable ({api_base_url}): {e}. "
            "Falling back to local execution."
        )
        
        # Fallback to local execution
        # Build a proper message list with context about file path if available
        ctx = dict(state.get("context", {}))
        messages_for_chain = list(state["messages"])
        
        # If there's a file_path in context, add it to the last message context
        if "file_path" in ctx and messages_for_chain:
            last_msg = messages_for_chain[-1]
            if hasattr(last_msg, 'content'):
                # Enhance the message with file path information
                file_info = f"\n\nNote: A report file has been created at {ctx['file_path']}. Please send this file as an attachment."
                if isinstance(last_msg, AIMessage):
                    # If last message is from DocumentCreator, create a new HumanMessage
                    from langchain_core.messages import HumanMessage
                    enhanced_msg = HumanMessage(
                        content=f"Send the report file {ctx['file_path']} via email to the recipient mentioned in the original request."
                    )
                    messages_for_chain.append(enhanced_msg)
                else:
                    # Enhance existing message
                    last_msg.content = last_msg.content + file_info
        
        chain = build_gmail_agent()
        result = chain.invoke({"messages": messages_for_chain})
        
        # Ensure result is an AIMessage with tool_calls preserved
        if not isinstance(result, AIMessage):
            if hasattr(result, 'content'):
                tool_calls = getattr(result, 'tool_calls', None)
                result = AIMessage(content=result.content, tool_calls=tool_calls if tool_calls else [])
            else:
                result = AIMessage(content=str(result))
        
        # Don't set capability here if there are tool calls
        # Let the Supervisor detect it from ToolMessage after execution
        if not getattr(result, "tool_calls", None):
            # Only set if no tool calls (direct response)
            content_lower = (result.content if hasattr(result, 'content') else str(result)).lower()
            if "email_sent" in content_lower:
                ctx["last_completed_capability"] = "send_email"
        
        return {
            "messages": [result],
            "context": ctx,
        }


register_agent(
    AgentSpec(
        name="Gmail",
        capabilities=[
            "gmail",
            "send_email",
            "search_email",
        ],
        build_chain=build_gmail_agent,
    )
)
