"""
Configuration for Gmail agent.

This module is fully self-contained.
It loads environment variables from the agent's local .env file
and resolves all paths relative to this agent directory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.logging_config import setup_logging

# Load .env from this agent directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
setup_logging()

# ===== OpenAI / LLM =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# ===== Gmail Configuration =====
GMAIL_CREDENTIALS_PATH = BASE_DIR / os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
GMAIL_TOKEN_PATH = BASE_DIR / os.getenv("GMAIL_TOKEN_PATH", "token.json")
GMAIL_SENDER_ADDRESS = os.getenv("GMAIL_SENDER_ADDRESS")

# ===== Safety checks (fail fast) =====
if not GMAIL_SENDER_ADDRESS:
    raise RuntimeError("GMAIL_SENDER_ADDRESS is not set in .env")

if not GMAIL_CREDENTIALS_PATH.exists():
    raise RuntimeError(f"Gmail credentials file not found: {GMAIL_CREDENTIALS_PATH}")

if not GMAIL_TOKEN_PATH.exists():
    raise RuntimeError(f"Gmail token file not found: {GMAIL_TOKEN_PATH}")
