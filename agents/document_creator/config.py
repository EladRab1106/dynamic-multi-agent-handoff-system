"""
Configuration for Document Creator agent.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.logging_config import setup_logging

load_dotenv()
setup_logging()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
