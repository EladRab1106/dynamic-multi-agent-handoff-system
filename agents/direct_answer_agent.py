import os
import requests
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import extract_completed_capability

from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Your task is to answer user questions directly and concisely. "
    "\n"
    "Instructions: "
    "1. Analyze the user's question. "
    "2. Provide a clear, direct answer. "
    "3. When your answer is COMPLETE, you MUST return ONLY a JSON object in this exact format: "
    '{{"completed_capability": "direct_answer"}}'
    "\n"
    "CRITICAL: "
    "- Do NOT return anything else after emitting the completion contract. "
    "- The completion contract must be the final and only output when the answer is done."
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
    
    if msg_type in ["human", "user"]:
        return HumanMessage(content=content)
    elif msg_type in ["ai", "assistant"]:
        return AIMessage(content=content)
    else:
        return AIMessage(content=content)


def build_direct_answer_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def direct_answer_node(state: AgentState):
    api_base_url = os.getenv("DIRECT_ANSWER_SERVICE_URL", "http://localhost:8003")
    
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in state")
    
    try:
        messages_dict = messages_to_dict(list(messages))
        
        response = requests.post(
            f"{api_base_url}/agent/invoke",
            json={
                "input": {
                    "messages": messages_dict
                }
            },
            timeout=30
        )
        response.raise_for_status()
        
        result_data = response.json()
        output = result_data.get("output")
        
        if isinstance(output, dict):
            try:
                if "type" in output or "content" in output:
                    result = message_from_dict(output)
                else:
                    result = AIMessage(content=str(output.get("content", output)))
            except Exception:
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
            f"direct answer creator Agent service is unavailable at {api_base_url}. "
            f"Remote execution is required. Original error: {e}"
        )


register_agent(
    AgentSpec(
        name="DirectAnswer",
        capabilities=["direct_answer"],
        build_chain=build_direct_answer_agent,
    )
)
