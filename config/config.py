# config/config.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

# Global LLM instance
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
