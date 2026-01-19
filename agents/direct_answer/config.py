"""
Configuration for Direct Answer agent.

Local copy to ensure the agent is fully self-contained.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:  # Prefer shared logging config if available
    from config.logging_config import setup_logging  # type: ignore
except Exception:  # Fallback for per-agent dev servers where root package isn't on sys.path
    def setup_logging() -> None:  # type: ignore[no-redef]
        """No-op logging setup used when shared config is unavailable."""
        return

load_dotenv()
setup_logging()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
