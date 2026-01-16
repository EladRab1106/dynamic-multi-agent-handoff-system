"""
Tools for Researcher agent.

Local copy to ensure the agent is fully self-contained.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set")

# Real Tavily tool
_real_tavily = TavilySearchResults(
    max_results=5
)

# Import tool usage tracker from researcher_agent
try:
    from researcher_agent import mark_tool_used
except ImportError:
    # Fallback if imported outside agent context
    def mark_tool_used():
        pass


@tool
def tavily_search(query: str) -> dict:
    """
    Search for information using Tavily search.
    
    Args:
        query: Search query string
    
    Returns:
        Search results from Tavily
    """
    mark_tool_used()
    return _real_tavily.invoke({"query": query})