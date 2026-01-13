"""
System test for Researcher Agent service.

Tests the Researcher agent via HTTP /agent/invoke endpoint.
Asserts that the agent returns a valid completion contract.
"""

import pytest
import requests
import json
import re
from typing import Dict, Any


RESEARCHER_SERVICE_URL = "http://localhost:8001"


def test_researcher_agent_invoke():
    """
    Test Researcher agent via HTTP /agent/invoke.
    
    Asserts:
    - HTTP 200 response
    - Valid JSON response
    - completed_capability == "research"
    - research_summary exists in data
    """
    # Prepare request payload
    # LangServe expects {"input": {"messages": [...]}}
    payload = {
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "Research information about Python programming language"
                }
            ]
        }
    }
    
    # Make HTTP request to agent service
    response = requests.post(
        f"{RESEARCHER_SERVICE_URL}/agent/invoke",
        json=payload,
        timeout=60  # Research may take time
    )
    
    # Assert HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Parse response
    result = response.json()
    assert "output" in result, "Response missing 'output' field"
    
    output = result["output"]
    
    # Handle different output formats (dict or string)
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
    assert contract["completed_capability"] == "research", f"Expected 'research', got '{contract['completed_capability']}'"
    
    # Assert research_summary exists
    assert "data" in contract, "Contract missing 'data' field"
    assert "research_summary" in contract["data"], "Contract missing 'data.research_summary'"
    
    research_summary = contract["data"]["research_summary"]
    assert isinstance(research_summary, dict), "research_summary must be a dictionary"
    
    # Assert research_summary has expected fields
    assert "topic" in research_summary or "summary" in research_summary, \
        "research_summary missing expected fields"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
