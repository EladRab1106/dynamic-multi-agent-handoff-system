from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user directly and concisely."
)


def build_direct_answer_agent():
    return create_agent(
        llm=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )


def direct_answer_node(state: AgentState):
    chain = build_direct_answer_agent()
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_completed_capability"] = "direct_answer"

    return {
        "messages": [result],
        "context": ctx,
    }


register_agent(
    AgentSpec(
        name="DirectAnswer",
        capabilities=["direct_answer"],
        build_chain=build_direct_answer_agent,
    )
)
