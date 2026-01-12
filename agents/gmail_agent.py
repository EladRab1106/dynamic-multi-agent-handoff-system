import os
import requests
from typing import Optional, List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import extract_completed_capability

from tools.gmail_tool import gmail_tool
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Gmail Agent. "
    "Your task is to send emails or search emails using the Gmail tool. "
    "\n"
    "Instructions: "
    "1. Use the Gmail tool to send emails or search emails as requested. "
    "2. If sending an email, you may optionally attach files (file paths must be real and available). "
    "3. When your task is COMPLETE, you MUST return ONLY a JSON object in this exact format: "
    "- For sending email: {{\"completed_capability\": \"send_email\"}}"
    "- For searching email: {{\"completed_capability\": \"search_email\", \"data\": {{\"results\": \"...\"}}}}"
    "\n"
    "CRITICAL: "
    "- Do NOT return anything else after emitting the completion contract. "
    "- The completion contract must be the final and only output when the task is done. "
    "- If the task is not complete, continue using tools and reasoning, but do NOT emit the contract yet."
)


def messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            msg_dict = msg.model_dump()
        elif hasattr(msg, 'dict'):
            msg_dict = msg.dict()
        else:
            msg_dict = {
                "type": msg.__class__.__name__.replace("Message", "").lower(),
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            }
        result.append(msg_dict)
    return result


def message_from_dict(msg_dict: Dict[str, Any]) -> BaseMessage:
    msg_type = msg_dict.get("type", "").lower()
    content = msg_dict.get("content", "")
    tool_calls = msg_dict.get("tool_calls", [])
    
    if msg_type in ["human", "user"]:
        return HumanMessage(content=content)
    elif msg_type in ["ai", "assistant"]:
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        else:
            return AIMessage(content=content)
    else:
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
    api_base_url = os.getenv("GMAIL_SERVICE_URL", "http://localhost:8000")
    
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in state")
    
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        message_content = last_message.content
    else:
        message_content = str(last_message)
    
    try:
        messages_dict = messages_to_dict(list(messages))
        
        response = requests.post(
            f"{api_base_url}/agent/invoke",
            json={
                "input": {
                    "messages": messages_dict
                }
            },
            timeout=5
        )
        response.raise_for_status()
        
        result_data = response.json()
        output = result_data.get("output")
        
        if output is None:
            result = AIMessage(content="Processing request...")
        elif isinstance(output, dict):
            content = output.get("content", str(output))
            result = AIMessage(content=content)
        elif isinstance(output, str):
            result = AIMessage(content=output)
        else:
            result = AIMessage(content=str(output))
        
        ctx = dict(state.get("context", {}))
        content = result.content if hasattr(result, 'content') else str(result)
        capability = extract_completed_capability(content)
        if capability:
            ctx["last_completed_capability"] = capability

        return {
            "messages": [result],
            "context": ctx,
        }

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Gmail Agent service is unavailable at {api_base_url}. "
            f"Remote execution is required. Original error: {e}"
        )


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
