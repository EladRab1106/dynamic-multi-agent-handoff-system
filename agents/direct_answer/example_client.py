"""
Example client for the Direct Answer Agent Service.

This demonstrates how to interact with the service from Python.
"""

import requests
from typing import Optional


class DirectAnswerAgentClient:
    """Client for interacting with the Direct Answer Agent Service."""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
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
    client = DirectAnswerAgentClient()
    
    print("=== Direct Answer Agent Service Client Examples ===\n")
    
    # Example: Direct answer
    print("1. Asking a direct question...")
    try:
        result = client.chat("What is the capital of France?")
        print(f"   Agent response: {result['response']}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
