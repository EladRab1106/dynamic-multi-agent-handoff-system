import os
from datetime import datetime
from agents.base_agent import create_agent
from langchain_core.messages import AIMessage
from config.config import llm
from models.state import AgentState

def document_creator_node(state: AgentState):
    os.makedirs("outputs", exist_ok=True)

    chain = create_agent(
        llm,
        [],
        "Convert the latest JSON research in the conversation into a clean Markdown report. "
        "Return ONLY Markdown with headings and bullet points."
    )

    md_msg = chain.invoke({"messages": state["messages"]})
    md_content = md_msg.content if hasattr(md_msg, "content") else str(md_msg)

    fp = f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(fp, "w") as f:
        f.write(md_content)

    ctx = dict(state.get("context", {}))
    ctx["file_path"] = fp
    ctx["last_task"] = "doc_created"

    return {
        "messages": [AIMessage(content=f"REPORT_CREATED: {fp}")],
        "context": ctx,
    }

