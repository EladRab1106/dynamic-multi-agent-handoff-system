from agents.base_agent import create_agent
from agents.registry import register_agent
from agents.spec import AgentSpec

from tools.search_tools import tavily
from config.config import llm
from models.state import AgentState


SYSTEM_PROMPT = (
    "You are the Researcher Agent. "
    "Use the Tavily search tool whenever you need external information. "
    "When finished, summarize your findings in JSON with keys: "
    "topic, summary, key_points, sources."
)


def build_researcher_agent():
    return create_agent(
        llm=llm,
        tools=[tavily],
        system_prompt=SYSTEM_PROMPT,
    )


def researcher_node(state: AgentState):
    chain = build_researcher_agent()
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_completed_capability"] = "research"

    return {
        "messages": [result],
        "context": ctx,
    }


register_agent(
    AgentSpec(
        name="Researcher",
        capabilities=["research", "web_search", "fact_lookup"],
        build_chain=build_researcher_agent,
    )
)
