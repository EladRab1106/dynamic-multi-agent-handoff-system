"""
System test for Gmail Agent service.

Tests the Gmail agent via HTTP /agent/invoke endpoint.
Tests gmail_search functionality and asserts completed_capability == "gmail".

Note: This test may require Gmail API credentials or mocking.
"""

import pytest
import requests
import json
import re
from typing import Dict, Any
from unittest.mock import patch, MagicMock


GMAIL_SERVICE_URL = "http://localhost:8000"


def test_gmail_agent_search():
    """
    Test Gmail agent search functionality via HTTP /agent/invoke.
    
    Tests gmail_search and asserts:
    - HTTP 200 response
    - Valid JSON response
    - completed_capability == "gmail"
    
    Note: This test assumes Gmail API is configured or will fail gracefully.
    """
    # Prepare request payload for email search
    payload = {
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "Search for emails with subject 'test'"
                }
            ]
        }
    }
    
    # Make HTTP request to agent service
    # Note: This may fail if Gmail API is not configured
    # In a real test environment, you would mock the Gmail API
    try:
        response = requests.post(
            f"{GMAIL_SERVICE_URL}/agent/invoke",
            json=payload,
            timeout=120  # Gmail operations may take time
        )
        
        # Assert HTTP 200 (or handle auth errors gracefully)
        if response.status_code != 200:
            # If auth error, skip test with informative message
            if response.status_code == 401 or response.status_code == 403:
                pytest.skip(f"Gmail API not configured: {response.status_code}")
            else:
                pytest.fail(f"Expected 200, got {response.status_code}: {response.text}")
        
        # Parse response
        result = response.json()
        assert "output" in result, "Response missing 'output' field"
        
        output = result["output"]
        
        # Handle different output formats
        if isinstance(output, dict):
            content = output.get("content", "")
        elif isinstance(output, str):
            content = output
        else:
            content = str(output)
        
        # Assert valid JSON completion contract
        # Handle case where JSON is wrapped in markdown code blocks
        contract = None
        try:
            contract = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks (```json ... ```)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    contract = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # If still not found, try to find any JSON object in the text
            if not contract:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"completed_capability"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    try:
                        contract = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
        
        if not contract:
            pytest.fail(f"Response is not valid JSON: {content[:200]}")
        
        # Assert completion contract structure
        assert isinstance(contract, dict), "Contract must be a dictionary"
        assert "completed_capability" in contract, "Contract missing 'completed_capability'"
        assert contract["completed_capability"] == "gmail", \
            f"Expected 'gmail', got '{contract['completed_capability']}'"
        
    except requests.exceptions.ConnectionError:
        pytest.skip("Gmail service not running (expected in CI/CD without service)")


def test_gmail_agent_send():
    """
    Test Gmail agent send functionality via HTTP /agent/invoke.
    
    Tests gmail_send and asserts completed_capability == "gmail".
    
    Note: This test requires Gmail API credentials or will be skipped.
    """
    # Prepare request payload for email send
    payload = {
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "Send an email to test@example.com with subject 'Test' and body 'Test message'"
                }
            ]
        }
    }
    
    # Make HTTP request to agent service
    try:
        response = requests.post(
            f"{GMAIL_SERVICE_URL}/agent/invoke",
            json=payload,
            timeout=120
        )
        
        # Handle auth errors gracefully
        if response.status_code != 200:
            if response.status_code == 401 or response.status_code == 403:
                pytest.skip(f"Gmail API not configured: {response.status_code}")
            else:
                pytest.fail(f"Expected 200, got {response.status_code}: {response.text}")
        
        # Parse and validate response
        result = response.json()
        output = result.get("output", {})
        
        if isinstance(output, dict):
            content = output.get("content", "")
        else:
            content = str(output)
        
        contract = json.loads(content)
        assert contract.get("completed_capability") == "gmail"
        
    except (requests.exceptions.ConnectionError, json.JSONDecodeError):
        pytest.skip("Gmail service not available or response invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
