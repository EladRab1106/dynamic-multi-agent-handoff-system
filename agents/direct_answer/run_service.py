#!/usr/bin/env python3
"""
Startup script for Direct Answer Agent Service.

Usage:
    python agents/direct_answer/run_service.py
    # or
    uvicorn agents.direct_answer.service:app --host 0.0.0.0 --port 8003
"""

import sys
import os
from pathlib import Path

# Add project root to Python path and change working directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "agents.direct_answer.service:app",
        host="0.0.0.0",
        port=8003,
        reload=True,  # Enable auto-reload for development
    )
