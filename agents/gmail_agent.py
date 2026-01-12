import os
import requests
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from tools.gmail_tool import gmail_tool
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Gmail Agent.\n"
    "Your responsibility is to handle ALL email-related tasks using the Gmail tool.\n\n"
    "You may search emails or send emails.\n\n"
    "IMPORTANT:\n"
    "- When you find an email, you MUST extract its FULL content.\n"
    "- Do NOT summarize unless explicitly asked.\n"
    "- Do NOT invent content.\n"
)


def messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        result.append({
            "type": msg.__class__.__name__.replace("Message", "").lower(),
            "content": msg.content if hasattr(msg, "content") else str(msg)
        })
    return result


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

    try:
        response = requests.post(
            f"{api_base_url}/agent/invoke",
            json={"input": {"messages": messages_to_dict(messages)}},
            timeout=20,
        )
        response.raise_for_status()

        output = response.json().get("output")

        ctx = dict(state.get("context", {}))

        # 🔑 אם Gmail Tool החזיר מייל – שומרים אותו
        email_data = None
        if isinstance(output, dict):
            email_data = output.get("email")

        ctx["gmail_data"] = {
            "email": email_data
        }

        ctx["last_completed_capability"] = "gmail"

        return {
            "messages": [
                AIMessage(content='{"completed_capability": "gmail"}')
            ],
            "context": ctx,
        }

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Gmail Agent service is unavailable at {api_base_url}. "
            f"Original error: {e}"
        )


register_agent(
    AgentSpec(
        name="Gmail",
        capabilities=["gmail"],
        build_chain=build_gmail_agent,
    )
)
