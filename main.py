from langchain_core.messages import HumanMessage
from models.state import AgentState
from graph.build_graph import build_graph


if __name__ == "__main__":
    print("=== Multi-Agent System (Context-Aware Supervisor + Planning) ===")

    q = input("Enter your request: ")


    workflow = build_graph()


    init_state: AgentState = {
        "messages": [HumanMessage(content=q)],
        "next": "Supervisor",
        "context": {},
    }

    for event in workflow.stream(init_state):
        if "__end__" in event:
            print("FINISHED.")
            break

        print(event)
        print("----")
