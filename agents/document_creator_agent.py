import os
from datetime import datetime

from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from langchain_core.messages import AIMessage
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a document creation agent. "
    "Convert the latest JSON research in the conversation into a clean Markdown report. "
    "Return ONLY Markdown with headings and bullet points."
)


def build_document_creator_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def document_creator_node(state: AgentState):
    os.makedirs("outputs", exist_ok=True)

    chain = build_document_creator_agent()
    md_msg = chain.invoke({"messages": state["messages"]})

    md_content = md_msg.content if hasattr(md_msg, "content") else str(md_msg)

    file_path = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    ctx = dict(state.get("context", {}))
    ctx["file_path"] = file_path
    ctx["last_completed_capability"] = "create_document"

    return {
        "messages": [AIMessage(content=f"REPORT_CREATED: {file_path}")],
        "context": ctx,
    }


register_agent(
    AgentSpec(
        name="DocumentCreator",
        capabilities=["create_document", "summarize", "write_document"],
        build_chain=build_document_creator_agent,
    )
)
