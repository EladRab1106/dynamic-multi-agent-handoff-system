from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
load_dotenv()


tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
