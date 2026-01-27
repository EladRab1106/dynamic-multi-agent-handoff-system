import logging
import os

from langchain_core.messages import HumanMessage
from models.state import AgentState
from graph.capability_discovery import discover_capabilities
from graph.build_graph import build_graph
from dotenv import load_dotenv
from config.logging_config import setup_logging

# Optional integration: aztm monkey-patches HTTP calls (e.g., for XMPP transport).
# Import early so any HTTP-based orchestration (e.g., RemoteGraph → Supervisor)
# is wrapped automatically when enabled in the environment.
try:
    import aztm  # type: ignore  # noqa: F401
except Exception:
    # Keep startup resilient if aztm is not installed; user can add it when needed.
    logging.getLogger(__name__).warning("aztm package not found; running without HTTP monkey patching")

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Ask for the user's request first so the CLI feels responsive,
    # even if AZTM/XMPP connection takes a while.
    

    # --- Step 1: Connect to AZTM transport ---
    from aztm.core.auth import login as aztm_login  # type: ignore
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    # Prefer explicit CLI user, then AZTM_USERID, then AZTM_JID from env (.env uses AZTM_JID)
    parser.add_argument(
        "--aztm-userid",
        dest="aztm_userid",
        default=os.environ.get("AZTM_USERID") or os.environ.get("AZTM_JID"),
    )
    parser.add_argument("--aztm-password", dest="aztm_password", default=os.environ.get("AZTM_PASSWORD"))
    parser.add_argument("--no-server-mode", dest="no_server_mode", action="store_true")
    args, _ = parser.parse_known_args()

    user = args.aztm_userid
    pwd = args.aztm_password
    server_mode = not args.no_server_mode

    if not user or not pwd:
        print("[startup] AZTM credentials are missing.")
        print("          Provide --aztm-userid/--aztm-password or set AZTM_USERID/AZTM_PASSWORD (or AZTM_JID).")
        raise SystemExit(1)

    print("=== Step 1/2: Connecting to AZTM transport ===")
    try:
        aztm.login(
            userid="rabiclient@connect.mishtamesh.net",
            password="clientpass",
            host="connect.mishtamesh.net",
            port=443,
            tls_mode="direct",
            validate_cert=False)
        logger.info("AZTM login successful for '%s' (server_mode=%s)", user, server_mode)
        print(f"[OK] Connected to AZTM as {user} (server_mode={server_mode})")
    except TimeoutError as e:
        logger.error("AZTM/XMPP connection timeout during login: %s", e, exc_info=True)
        print("[ERROR] External: timed out connecting to AZTM/XMPP.")
        print("        Check AZTM_HOST, AZTM_PORT, TLS settings, and network connectivity.")
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        logger.error("AZTM/XMPP login failed: %s", e, exc_info=True)
        print("[ERROR] External: AZTM/XMPP login failed.")
        print("        This is an AZTM or network configuration issue, not a Supervisor code bug.")
        raise SystemExit(1)
    q = input("Enter your request: ")
    # --- Existing supervisor-only / multi-agent selection ---
    # Default to supervisor-only mode unless explicitly disabled.
    # SUPERVISOR_ONLY_MODE values treated as FALSE: "0", "false", "no" (case-insensitive).
    _supervisor_only_raw = os.getenv("SUPERVISOR_ONLY_MODE")
    if _supervisor_only_raw is None:
        supervisor_only = True
    else:
        supervisor_only = _supervisor_only_raw.lower() not in {"0", "false", "no"}

    if supervisor_only:
        print("=== Supervisor-Only Mode (no worker agents) ===")
        logger.info("Starting in supervisor-only mode: skipping capability discovery")
        capability_index = {}
        capabilities = []

        print("\n" + "=" * 60)
        print("SUPERVISOR-ONLY MODE")
        print("=" * 60)
        print("Running with Supervisor only. No worker capabilities will be used.")
        print("All requests will be handled directly by the Supervisor agent.\n")
    else:
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
    print("=== Step 2/2: Running Supervisor workflow ===")

    # Use invoke() for blocking execution - ensures complete termination
    # This avoids streaming deadlocks and guarantees final state
    try:
        final_state = workflow.invoke(init_state)
        logger.info("Workflow execution completed successfully")
        print("[OK] Workflow completed.")
        if "messages" in final_state and final_state["messages"]:
            last_msg = final_state["messages"][-1]
            if hasattr(last_msg, "content"):
                print(f"Final message (truncated): {last_msg.content[:200]}...")
    except RuntimeError as e:
        msg = str(e)
        if "Not logged in. Call aztm.login() first" in msg:
            logger.error("AZTM HTTP hook reports missing login: %s", e, exc_info=True)
            print("[ERROR] External: AZTM HTTP/XMPP hook is not logged in.")
            print("        Ensure the AZTM login step at startup succeeded.")
        else:
            logger.error("Workflow execution failed (RuntimeError): %s", e, exc_info=True)
            print("[ERROR] Internal: Supervisor/graph workflow failed (RuntimeError).")
            print(f"        Details: {msg}")
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        logger.error("Workflow execution failed (unexpected): %s", e, exc_info=True)
        print("[ERROR] Internal: Supervisor/graph workflow failed (unexpected error).")
        print(f"        Details: {e}")
        raise SystemExit(1)
