from agents.base_agent import create_agent
from tools.search_tools import tavily
from config.config import llm
from models.state import AgentState

def researcher_node(state: AgentState):
    chain = create_agent(
        llm,
        [tavily],
        (
            "You are the Researcher Agent. "
            "Use the Tavily search tool whenever you need external information. "
            "When finished, summarize your findings in JSON with keys: "
            "topic, summary, key_points, sources."
        ),
    )
    result = chain.invoke({"messages": state["messages"]})

    ctx = dict(state.get("context", {}))
    ctx["last_task"] = "research"

    return {
        "messages": [result],
        "context": ctx,
    }