from agents.base_agent import create_agent
from config.config import llm
from models.state import AgentState


def direct_answer_node(state: AgentState):
    chain = create_agent(
        llm,
        [],
        "You are a helpful assistant. Answer the user directly."
    )
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_task"] = "direct_answer"

    return {
        "messages": [result],
        "context": ctx,
    }
