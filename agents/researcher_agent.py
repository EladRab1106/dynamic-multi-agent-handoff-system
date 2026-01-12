import os
import requests
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import extract_completed_capability

from tools.search_tools import tavily
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Researcher Agent. "
    "Your task is to research topics using the Tavily search tool. "
    "\n"
    "Instructions: "
    "1. Use the Tavily search tool to gather information about the requested topic. "
    "2. Analyze and summarize your findings. "
    "3. When your research is COMPLETE, you MUST return ONLY a JSON object in this exact format: "
    '{{"completed_capability": "research", "data": {{"research_summary": {{"topic": "...", "summary": "...", "key_points": [...], "sources": [...]}}}}}}'
    "\n"
    "CRITICAL: "
    "- Do NOT return anything else after emitting the completion contract. "
    "- The completion contract must be the final and only output when research is done. "
    "- If research is not complete, continue using tools and reasoning, but do NOT emit the contract yet."
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


def build_researcher_agent():
    return create_agent(
        llm=llm,
        tools=[tavily],
        system_prompt=SYSTEM_PROMPT,
    )


def researcher_node(state: AgentState):
    api_base_url = os.getenv("RESEARCHER_SERVICE_URL", "http://localhost:8001")
    
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
            timeout=60
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
            f"researcher creator Agent service is unavailable at {api_base_url}. "
            f"Remote execution is required. Original error: {e}"
        )


register_agent(
    AgentSpec(
        name="Researcher",
        capabilities=["research"],
        build_chain=build_researcher_agent,
    )
)
