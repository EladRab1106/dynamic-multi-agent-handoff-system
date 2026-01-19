import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.logging_config import setup_logging

load_dotenv()
setup_logging()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
