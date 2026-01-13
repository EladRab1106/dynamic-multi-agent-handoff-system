"""
System test for Direct Answer Agent service.

Tests the Direct Answer agent via HTTP /agent/invoke endpoint.
Asserts that the agent returns a valid completion contract.
"""

import pytest
import requests
import json
import re
from typing import Dict, Any


DIRECT_ANSWER_SERVICE_URL = "http://localhost:8003"


def test_direct_answer_agent_invoke():
    """
    Test Direct Answer agent via HTTP /agent/invoke.
    
    Sends a simple question and asserts:
    - HTTP 200 response
    - Valid JSON response
    - completed_capability == "direct_answer"
    """
    # Prepare request payload with a simple question
    payload = {
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "What is 2 + 2?"
                }
            ]
        }
    }
    
    # Make HTTP request to agent service
    response = requests.post(
        f"{DIRECT_ANSWER_SERVICE_URL}/agent/invoke",
        json=payload,
        timeout=30
    )
    
    # Assert HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
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
    assert contract["completed_capability"] == "direct_answer", \
        f"Expected 'direct_answer', got '{contract['completed_capability']}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
