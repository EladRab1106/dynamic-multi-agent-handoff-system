from langgraph.graph import StateGraph, END
from agents.researcher_agent import researcher_node
from agents.gmail_agent import gmail_node
from agents.document_creator_agent import document_creator_node
from agents.direct_answer_agent import direct_answer_node
from supervisor.supervisor import supervisor_node
from models.state import AgentState

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Researcher", researcher_node)
    graph.add_node("DocumentCreator", document_creator_node)
    graph.add_node("Gmail", gmail_node)
    graph.add_node("DirectAnswer", direct_answer_node)

    # All agents now return directly to Supervisor after completing their work
    graph.add_edge("Researcher", "Supervisor")
    graph.add_edge("DocumentCreator", "Supervisor")
    graph.add_edge("Gmail", "Supervisor")
    graph.add_edge("DirectAnswer", "Supervisor")

    graph.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "DocumentCreator": "DocumentCreator",
            "Gmail": "Gmail",
            "DirectAnswer": "DirectAnswer",
            "FINISH": END,
        }
    )

    graph.set_entry_point("Supervisor")

    return graph.compile()
