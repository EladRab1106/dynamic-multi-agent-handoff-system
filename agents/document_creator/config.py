"""
Configuration for Document Creator agent.
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

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
