"""
Run Researcher agent as LangGraph Server.

This agent should be run as an independent LangGraph Server instance.
Each agent can live in its own repository and be deployed independently.

To run this agent:
    langgraph dev --port 8001

Or in production:
    langgraph serve --port 8001

The agent exposes its graph via LangGraph Server's native endpoints:
- POST /runs - Execute graph runs
- GET /assistants - List assistants (for capability discovery)
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

if __name__ == "__main__":
    print("=" * 60)
    print("Researcher Agent - LangGraph Server")
    print("=" * 60)
    print("\nTo run this agent, use LangGraph CLI:")
    print("  langgraph dev --port 8001")
    print("\nOr for production:")
    print("  langgraph serve --port 8001")
    print("\nThe agent will be accessible at:")
    print("  http://localhost:8001")
    print("\nGraph ID: researcher")
    print("=" * 60)
