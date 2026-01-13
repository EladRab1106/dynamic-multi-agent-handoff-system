"""
Example client for the Gmail Agent Service.

This demonstrates how to interact with the service from Python.
"""

import requests
from typing import Optional, List


class GmailAgentClient:
    """Client for interacting with the Gmail Agent Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None
    ) -> dict:
        """Send an email via the direct endpoint."""
        response = requests.post(
            f"{self.base_url}/send-email",
            json={
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "attachments": attachments or []
            }
        )
        response.raise_for_status()
        return response.json()
    
    def read_email(self, query: str) -> dict:
        """Search/read emails via the direct endpoint."""
        response = requests.post(
            f"{self.base_url}/read-email",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    
    def chat(self, message: str) -> dict:
        """Chat with the agent using natural language."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={"message": message}
        )
        response.raise_for_status()
        return response.json()
    
    def invoke_agent(self, messages: List[dict]) -> dict:
        """Invoke the agent via LangServe endpoint."""
        response = requests.post(
            f"{self.base_url}/agent/invoke",
            json={"input": {"messages": messages}}
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Example usage
    client = GmailAgentClient()
    
    print("=== Gmail Agent Service Client Examples ===\n")
    
    # Example 1: Send an email
    print("1. Sending an email...")
    try:
        result = client.send_email(
            recipient="test@example.com",
            subject="Test Email",
            body="This is a test email from the API client."
        )
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Example 2: Search emails
    print("2. Searching emails...")
    try:
        result = client.read_email(query="from:example@gmail.com")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Example 3: Chat with agent
    print("3. Chatting with agent...")
    try:
        result = client.chat("Find emails from last week")
        print(f"   Agent response: {result['response']}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
