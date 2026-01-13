"""
Example client for the Document Creator Agent Service.

This demonstrates how to interact with the service from Python.
"""

import requests
from typing import Optional


class DocumentCreatorAgentClient:
    """Client for interacting with the Document Creator Agent Service."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip("/")
    
    def chat(self, message: str) -> dict:
        """Chat with the agent using natural language."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={"message": message}
        )
        response.raise_for_status()
        return response.json()
    
    def invoke_agent(self, messages: list) -> dict:
        """Invoke the agent via LangServe endpoint."""
        response = requests.post(
            f"{self.base_url}/agent/invoke",
            json={"input": {"messages": messages}}
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Example usage
    client = DocumentCreatorAgentClient()
    
    print("=== Document Creator Agent Service Client Examples ===\n")
    
    # Example: Create document
    print("1. Creating a document...")
    try:
        result = client.chat("Create a markdown report from the research data")
        print(f"   Agent response: {result['response']}")
        print(f"   File path: {result.get('file_path', 'N/A')}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
