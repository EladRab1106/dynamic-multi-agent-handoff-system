"""
Configuration for Document Creator agent.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:  # Prefer shared logging config if available
    from config.logging_config import setup_logging  # type: ignore
except Exception:  # Fallback for per-agent dev servers where root package isn't on sys.path
    import logging

    def setup_logging() -> None:  # type: ignore[no-redef]
        """Basic logging setup used when shared config is unavailable.

        This ensures agents running standalone (e.g. in Cloud Run) emit INFO-level
        logs to stdout/stderr so that file creation and error details are visible.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )

load_dotenv()
setup_logging()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
