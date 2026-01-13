import os
import requests
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec
from agents.utils import (
    build_completion_message,
    parse_completion_message,
    extract_completed_capability,
)

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
        
        # Parse completion contract using shared utility
        contract = parse_completion_message(content)
        
        email_data = None
        if contract:
            # Extract email data from completion contract
            if "data" in contract and "email" in contract["data"]:
                email_data = contract["data"]["email"]
            
            # Extract completed capability using shared utility
            capability = extract_completed_capability(content)
            if capability:
                ctx["last_completed_capability"] = capability
        
        # Store email data in generic document_source contract
        # This allows DocumentCreator to work with any source type
        if email_data:
            ctx["document_source"] = {
                "type": "email",
                "content": email_data
            }
        
        # If no valid completion contract was found, create one using shared utility
        if not contract:
            completion_data = {"email": email_data} if email_data else {"email": None}
            result = build_completion_message("gmail", completion_data)
            # Extract capability from the contract we just created (fully contract-driven)
            capability = extract_completed_capability(result.content)
            if capability:
                ctx["last_completed_capability"] = capability
        # If contract exists, the agent already returned a valid completion contract
        # The result is already correct, and capability was set above

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
