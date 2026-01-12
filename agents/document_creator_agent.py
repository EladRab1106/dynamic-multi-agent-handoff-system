import os
import requests
from datetime import datetime
from typing import List, Dict, Any

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import extract_completed_capability, parse_completion_message

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a document creation agent. "
    "Your task is to convert research data into a clean Markdown report. "
    "\n"
    "Instructions: "
    "1. Find the latest research data in the conversation (JSON format with research_summary). "
    "2. Convert it into a well-structured Markdown report with headings and bullet points. "
    "3. When the document is COMPLETE, you MUST return ONLY a JSON object in this exact format: "
    '{{"completed_capability": "create_document", "data": {{"markdown": "# Report\\n\\n...", "file_path": "outputs/report_YYYYMMDD_HHMMSS.md"}}}}'
    "\n"
    "CRITICAL: "
    "- The file_path must follow the pattern: outputs/report_YYYYMMDD_HHMMSS.md "
    "- Do NOT return anything else after emitting the completion contract. "
    "- The completion contract must be the final and only output when the document is done. "
    "- Include the full markdown content in the 'markdown' field of the data object."
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


def build_document_creator_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def document_creator_node(state: AgentState):
    api_base_url = os.getenv("DOCUMENT_CREATOR_SERVICE_URL", "http://localhost:8002")
    
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
        
        content = result.content if hasattr(result, "content") else str(result)
        contract = parse_completion_message(content)
        
        ctx = dict(state.get("context", {}))
        
        if contract and "data" in contract:
            data = contract["data"]
            
            if "markdown" in data:
                os.makedirs("outputs", exist_ok=True)
                file_path = data.get("file_path") or f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(data["markdown"])
                ctx["file_path"] = file_path
            elif "file_path" in data:
                ctx["file_path"] = data["file_path"]
            
            capability = contract.get("completed_capability")
            if capability:
                ctx["last_completed_capability"] = capability
        
        return {
            "messages": [result],
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
