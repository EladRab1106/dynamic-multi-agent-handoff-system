import logging
from langchain_core.messages import HumanMessage
from models.state import AgentState
from graph.capability_discovery import discover_capabilities
from graph.build_graph import build_graph
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    print("=== Multi-Agent System (Context-Aware Supervisor + Planning) ===")
    
    # Discover agent capabilities BEFORE building the graph
    # Capability index is used by orchestrator for routing (capability → agent_name)
    # Capabilities list is injected into Supervisor for planning
    logger.info("Starting capability discovery...")
    try:
        capability_index = discover_capabilities()
        capabilities = sorted(capability_index.keys())
        
        # Print discovered capabilities in readable format
        print("\n" + "=" * 60)
        print("DISCOVERED CAPABILITIES:")
        print("=" * 60)
        if capability_index:
            for capability, agent_name in sorted(capability_index.items()):
                print(f"✔ {capability} → {agent_name}")
        else:
            print("❌ No capabilities discovered")
        print("=" * 60 + "\n")
        
        if not capability_index:
            print("❌ Startup aborted — no agent capabilities available")
            raise RuntimeError("No agent capabilities discovered")
        
        logger.info(f"Capability discovery complete: {len(capability_index)} capabilities registered")
    except Exception as e:
        logger.error(f"Capability discovery failed: {e}")
        print(f"\n❌ Startup aborted: {e}\n")
        raise

    q = input("Enter your request: ")

    # Build graph with capability_index for routing (capability → agent_name)
    workflow = build_graph(capability_index=capability_index)

    # Inject ONLY capabilities into context for Supervisor
    # Supervisor plans using capability strings only
    # Orchestrator maps capability → agent_name for routing
    init_state: AgentState = {
        "messages": [HumanMessage(content=q)],
        "next": "Supervisor",
        "context": {
            "capabilities": capabilities,  # Only capability strings, no agent names
        },
    }

    logger.info("Starting workflow execution...")
    
    # Use invoke() for blocking execution - ensures complete termination
    # This avoids streaming deadlocks and guarantees final state
    try:
        final_state = workflow.invoke(init_state)
        logger.info("Workflow execution completed successfully")
        print("\n=== Workflow Completed ===")
        print(f"Final state keys: {list(final_state.keys())}")
        if "messages" in final_state and final_state["messages"]:
            last_msg = final_state["messages"][-1]
            if hasattr(last_msg, 'content'):
                print(f"Final message: {last_msg.content[:200]}...")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        print(f"\n❌ Workflow failed: {e}")
        raise
