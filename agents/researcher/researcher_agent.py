import os
import requests
import json
import re
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import (
    build_completion_message,
    parse_completion_message,
    extract_completed_capability,
)

from tools.search_tools import tavily
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Researcher Agent.\n"
    "Your ONLY task is to research topics using the Tavily search tool.\n\n"

    "INSTRUCTIONS:\n"
    "1. Use the Tavily search tool to gather information about the requested topic.\n"
    "2. Analyze and summarize your findings.\n"
    "3. When your research is COMPLETE, you MUST return ONLY valid JSON in this exact format:\n"
    "The JSON should have 'completed_capability' set to 'research' and 'data.research_summary' containing 'topic', 'summary', 'key_points' (list), and 'sources' (list).\n\n"

    "CRITICAL RULES:\n"
    "- Return ONLY the JSON completion contract - no other text, no explanations, no markdown\n"
    "- Do NOT mention other agents' responsibilities (e.g., sending emails, creating documents)\n"
    "- Do NOT say 'I will now...' or 'I cannot...' - just return the JSON\n"
    "- Start your response with a JSON object and end with a JSON object\n"
    "- The completion contract must be the final and only output when research is done\n"
    "- If research is not complete, continue using tools - do NOT emit the contract yet\n"
    "- Focus ONLY on research - do not discuss what happens next"
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


def _extract_research_data_from_text(content: str) -> Dict[str, Any] | None:
    """
    Attempt to extract research data from plain text response.
    
    This is a fallback mechanism when the agent doesn't return proper JSON.
    Tries to find structured data in the text that might represent research results.
    """
    if not content:
        return None
    
    # Try to find JSON-like structures in the text
    # Look for patterns like {"topic": "...", "summary": "..."}
    try:
        # Try to parse as JSON first
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "research_summary" in parsed.get("data", {}):
            return parsed["data"]["research_summary"]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict) and "research_summary" in parsed.get("data", {}):
                return parsed["data"]["research_summary"]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Try to find any JSON object in the text
    json_match = re.search(r'\{[^{}]*"research_summary"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and "research_summary" in parsed.get("data", {}):
                return parsed["data"]["research_summary"]
        except (json.JSONDecodeError, TypeError):
            pass
    
    return None


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
        
        # Parse completion contract using shared utility
        contract = parse_completion_message(content)
        
        research_data = None
        if contract:
            # Extract research data from completion contract
            if "data" in contract and "research_summary" in contract["data"]:
                research_data = contract["data"]["research_summary"]
            
            # Extract completed capability using shared utility
            capability = extract_completed_capability(content)
            if capability:
                ctx["last_completed_capability"] = capability
        else:
            # Contract parsing failed - agent may have returned plain text
            # Try to extract research data from the response and force proper completion format
            # This is a safety mechanism to prevent infinite loops
            research_data = _extract_research_data_from_text(content)
            
            # Force proper completion contract using shared utility
            if research_data:
                completion_data = {"research_summary": research_data}
            else:
                # Fallback: create minimal research summary from content
                research_data = {
                    "topic": "Unknown",
                    "summary": content[:500] if len(content) > 500 else content,
                    "key_points": [],
                    "sources": []
                }
                completion_data = {"research_summary": research_data}
            
            result = build_completion_message("research", completion_data)
            # Extract capability from the contract we just created (fully contract-driven)
            capability = extract_completed_capability(result.content)
            if capability:
                ctx["last_completed_capability"] = capability
        
        # Store research data in generic document_source contract
        # This allows DocumentCreator to work with any source type
        if research_data:
            ctx["document_source"] = {
                "type": "research",
                "content": research_data
            }
        
        # Debug assertion: verify completion contract is valid
        # This catches contract violations early in development
        final_content = result.content if hasattr(result, 'content') else str(result)
        final_contract = parse_completion_message(final_content)
        if not final_contract:
            raise ValueError(
                f"Researcher agent failed to return valid completion contract. "
                f"Content: {final_content[:200]}..."
            )
        
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
