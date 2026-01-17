"""
Configuration for Document Creator agent.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing"

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
