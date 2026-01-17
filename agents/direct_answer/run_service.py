"""
Run Direct Answer agent as LangGraph Server.

This agent should be run as an independent LangGraph Server instance.
Each agent can live in its own repository and be deployed independently.

To run this agent:
    langgraph dev --port 8003

Or in production:
    langgraph serve --port 8003

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
    print("Direct Answer Agent - LangGraph Server")
    print("=" * 60)
    print("\nTo run this agent, use LangGraph CLI:")
    print("  langgraph dev --port 8003")
    print("\nOr for production:")
    print("  langgraph serve --port 8003")
    print("\nThe agent will be accessible at:")
    print("  http://localhost:8003")
    print("\nGraph ID: direct_answer")
    print("=" * 60)
