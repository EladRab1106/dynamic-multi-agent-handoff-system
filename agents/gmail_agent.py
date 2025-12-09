from agents.base_agent import create_agent
from tools.gmail_tool import gmail_tool
from config.config import llm
from models.state import AgentState

def gmail_node(state: AgentState):
    ctx = dict(state.get("context", {}))

    chain = create_agent(
        llm,
        [gmail_tool],
        (
    "You are the Gmail Agent.\n"
    "Your job is to understand the user's intent regarding email actions.\n"
    "- Decide whether the user wants to SEARCH emails or SEND an email.\n"
    "- If SEARCH: infer an appropriate Gmail query.\n"
    "- If SEND: extract recipient, subject, and body.\n"
    "- When the user asks to send the body of a previously found email, "
    "you MUST use the actual text returned by the search (the snippet or full body) "
    "and include it in the send action.\n"
    "- Ask clarifying questions if any field is missing.\n"
    "When you are ready, call the `gmail_tool` with the correct parameters.\n"
)
,
    )

    result = chain.invoke({"messages": state["messages"]})

    ctx["last_task"] = "gmail"

    return {
        "messages": [result],
        "context": ctx,
    }