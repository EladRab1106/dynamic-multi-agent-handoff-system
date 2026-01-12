import os
import json
from datetime import datetime

from langchain_core.messages import AIMessage
from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a document creation agent.\n"
    "You MUST create a document ONLY from the data provided to you.\n"
    "You are STRICTLY FORBIDDEN from inventing or assuming content.\n\n"
    "If no data is provided, you MUST explicitly say so.\n\n"
    "When done, return ONLY:\n"
    '{ "completed_capability": "create_document", "data": { "markdown": "...", "file_path": "..." } }'
)


def build_document_creator_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def document_creator_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    gmail_data = ctx.get("gmail_data", {})
    email = gmail_data.get("email")

    # ❗ אין מייל → לא ממציאים
    if not email:
        content = json.dumps({
            "completed_capability": "create_document",
            "data": {
                "markdown": "No email content was found.",
                "file_path": None
            }
        })

        ctx["last_completed_capability"] = "create_document"
        return {
            "messages": [AIMessage(content=content)],
            "context": ctx
        }

    markdown = f"""# Email Report

**From:** {email.get("from")}
**Subject:** {email.get("subject")}
**Date:** {email.get("date")}

---

{email.get("body")}
"""

    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    ctx["file_path"] = file_path
    ctx["last_completed_capability"] = "create_document"

    content = json.dumps({
        "completed_capability": "create_document",
        "data": {
            "markdown": markdown,
            "file_path": file_path
        }
    })

    return {
        "messages": [AIMessage(content=content)],
        "context": ctx
    }


register_agent(
    AgentSpec(
        name="DocumentCreator",
        capabilities=["create_document"],
        build_chain=build_document_creator_agent,
    )
)
