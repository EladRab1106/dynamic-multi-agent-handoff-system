import os
import requests
import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from tools.gmail_search_tool import gmail_search
from tools.gmail_send_tool import gmail_send
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Gmail Agent.\n"
    "You have access to two tools:\n"
    "1. gmail_search - Search for emails by query\n"
    "   Returns JSON: {{\"email\": {{\"from\": \"...\", \"subject\": \"...\", \"date\": \"...\", \"body\": \"...\"}}}} or {{\"email\": null}}\n"
    "2. gmail_send - Send emails\n"
    "   Returns JSON: {{\"status\": \"sent\", \"to\": \"...\", \"subject\": \"...\"}}\n\n"
    "WORKFLOW FOR SEARCHING:\n"
    "1. Call gmail_search with the search query\n"
    "2. The tool returns a JSON object with email data\n"
    "3. Extract the email object from the tool result\n"
    "4. Return completion JSON with the email data included:\n"
    "   {{\"completed_capability\": \"gmail\", \"data\": {{\"email\": <copy the email object from tool result>}}}}\n"
    "5. If tool returns {{\"email\": null}}, return: {{\"completed_capability\": \"gmail\", \"data\": {{\"email\": null}}}}\n\n"
    "WORKFLOW FOR SENDING:\n"
    "1. Call gmail_send with recipient, subject, body, attachments\n"
    "2. Return: {{\"completed_capability\": \"gmail\", \"data\": {{\"status\": \"sent\"}}}}\n\n"
    "CRITICAL RULES:\n"
    "- When searching, you MUST include the full email object from gmail_search in your completion JSON\n"
    "- Copy the email object exactly as returned by gmail_search tool\n"
    "- Return ONLY the completion JSON - no other text, no code blocks, no markdown\n"
    "- Use proper JSON syntax with single braces\n"
    "- The email object must have: from, subject, date, body fields"
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
        tools=[gmail_search, gmail_send],
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
            timeout=120,
        )
        response.raise_for_status()

        result_data = response.json()
        output = result_data.get("output")

        if isinstance(output, dict):
            result = AIMessage(content=output.get("content", str(output)))
        elif isinstance(output, str):
            result = AIMessage(content=output)
        else:
            result = AIMessage(content=str(output))

        ctx = dict(state.get("context", {}))

        content = result.content if hasattr(result, 'content') else str(result)
        email_data = None
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if "data" in parsed and "email" in parsed["data"]:
                    email_data = parsed["data"]["email"]
                if parsed.get("completed_capability") == "gmail":
                    ctx["last_completed_capability"] = "gmail"
        except (json.JSONDecodeError, TypeError):
            pass
        
        ctx["gmail_data"] = {
            "email": email_data
        }
        
        # Ensure completion contract is set and properly formatted
        if "last_completed_capability" not in ctx:
            # Create proper completion contract with email data if available
            completion_data = {"completed_capability": "gmail"}
            if email_data:
                completion_data["data"] = {"email": email_data}
            else:
                completion_data["data"] = {"email": None}
            result = AIMessage(content=json.dumps(completion_data))
            ctx["last_completed_capability"] = "gmail"
        else:
            # Ensure result has proper completion contract format
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict) or "completed_capability" not in parsed:
                    # Rebuild with proper format
                    completion_data = {"completed_capability": "gmail"}
                    if email_data:
                        completion_data["data"] = {"email": email_data}
                    else:
                        completion_data["data"] = {"email": None}
                    result = AIMessage(content=json.dumps(completion_data))
            except:
                # Content is not JSON, rebuild
                completion_data = {"completed_capability": "gmail"}
                if email_data:
                    completion_data["data"] = {"email": email_data}
                else:
                    completion_data["data"] = {"email": None}
                result = AIMessage(content=json.dumps(completion_data))

        return {
            "messages": [result],
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
