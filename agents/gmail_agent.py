from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from tools.gmail_tool import gmail_tool
from config.config import llm
from models.state import AgentState

SYSTEM_PROMPT = (
    "You are the Gmail Agent.\n"
    "You can SEARCH emails or SEND emails.\n"
    "- If sending an email, you may optionally attach files.\n"
    "- Attachments must be real file paths available in the system.\n"
    "- Only attach files if the user explicitly asks for it.\n"
    "- If unsure which file to attach, ask a clarification question.\n"
)


def build_gmail_agent():
    return create_agent(
        llm=llm,
        tools=[gmail_tool],
        system_prompt=SYSTEM_PROMPT,
    )


def gmail_node(state: AgentState):
    chain = build_gmail_agent()
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))

    if getattr(result, "tool_calls", None):
        ctx["last_completed_capability"] = "send_email"

    return {
        "messages": [result],
        "context": ctx,
    }


register_agent(
    AgentSpec(
        name="Gmail",
        capabilities=[
            "gmail",
            "send_email",
            "search_email",
        ],
        build_chain=build_gmail_agent,
    )
)
