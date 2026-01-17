"""
Configuration for Supervisor agent.

Local copy to ensure the agent is fully self-contained.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
