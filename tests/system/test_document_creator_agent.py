"""
System test for Document Creator Agent service.

Tests the Document Creator agent via HTTP /agent/invoke endpoint.
Asserts that the agent returns a valid completion contract with markdown and file_path.
"""

import pytest
import requests
import json
import re
from typing import Dict, Any


DOCUMENT_CREATOR_SERVICE_URL = "http://localhost:8002"


def test_document_creator_agent_invoke():
    """
    Test Document Creator agent via HTTP /agent/invoke.
    
    Sends a document_source payload and asserts:
    - HTTP 200 response
    - Valid JSON response
    - completed_capability == "create_document"
    - markdown exists in data
    - file_path exists in data
    """
    # Prepare request payload with document_source context
    # The agent should read from context, but we'll send it via messages
    # Note: In real flow, document_source comes from context, but for testing
    # we simulate it via a message that the agent can process
    payload = {
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "Create a document from the following research data: "
                               '{"topic": "Test Topic", "summary": "Test summary", "key_points": ["Point 1"], "sources": []}'
                }
            ]
        }
    }
    
    # Make HTTP request to agent service
    response = requests.post(
        f"{DOCUMENT_CREATOR_SERVICE_URL}/agent/invoke",
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
    assert contract["completed_capability"] == "create_document", \
        f"Expected 'create_document', got '{contract['completed_capability']}'"
    
    # Assert data fields exist
    assert "data" in contract, "Contract missing 'data' field"
    data = contract["data"]
    
    assert "markdown" in data, "Contract missing 'data.markdown'"
    assert isinstance(data["markdown"], str), "markdown must be a string"
    assert len(data["markdown"]) > 0, "markdown must not be empty"
    
    assert "file_path" in data, "Contract missing 'data.file_path'"
    assert isinstance(data["file_path"], str), "file_path must be a string"
    assert data["file_path"].startswith("outputs/"), "file_path should start with 'outputs/'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
