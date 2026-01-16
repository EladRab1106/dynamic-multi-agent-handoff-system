"""
Utility functions for Researcher agent.

Local copy of contract parsing utilities to ensure the agent is fully self-contained.
"""

import json
from typing import Dict, Any, Optional


def parse_completion_message(message_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a completion contract from message content.
    
    Tries to extract JSON even if embedded in text.
    Returns:
        Dict with completion contract if valid, None otherwise
    """
    if not message_content:
        return None

    # First try: parse entire content as JSON
    try:
        parsed = json.loads(message_content.strip())
        if isinstance(parsed, dict) and "completed_capability" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Second try: find JSON object boundaries and extract
    content_stripped = message_content.strip()
    json_start = content_stripped.find('{')
    json_end = content_stripped.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_content = content_stripped[json_start:json_end+1]
            parsed = json.loads(json_content)
            if isinstance(parsed, dict) and "completed_capability" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def extract_completed_capability(message_content: str) -> Optional[str]:
    """
    Extract the completed capability from a completion contract.
    
    Returns:
        Capability string if found, None otherwise
    """
    contract = parse_completion_message(message_content)
    if contract:
        return contract.get("completed_capability")
    return None


def validate_completion_contract_strict(content: str) -> Dict[str, Any]:
    """
    Strictly validate that content contains ONLY a valid completion contract JSON.
    
    Raises ValueError if:
    - Content is not valid JSON
    - Content contains extra text outside JSON
    - JSON is not a valid completion contract
    
    Returns the parsed contract if valid.
    """
    if not content:
        raise ValueError("Completion contract content is empty")
    
    content_stripped = content.strip()
    if not content_stripped:
        raise ValueError("Completion contract content is empty after stripping")
    
    # Find JSON boundaries
    json_start = content_stripped.find('{')
    json_end = content_stripped.rfind('}')
    
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise ValueError(
            f"Completion contract does not contain valid JSON object. "
            f"Content: {content[:500]}..."
        )
    
    # Check for text before or after JSON
    before_json = content_stripped[:json_start].strip()
    after_json = content_stripped[json_end+1:].strip()
    
    if before_json or after_json:
        raise ValueError(
            f"Completion contract contains extra text outside JSON. "
            f"Text before JSON: '{before_json}', Text after JSON: '{after_json}'. "
            f"Content must contain ONLY valid JSON. Full content: {content[:500]}..."
        )
    
    # Extract and parse JSON
    json_content = content_stripped[json_start:json_end+1]
    
    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Completion contract is not valid JSON: {e}. "
            f"Content: {content[:500]}..."
        )
    
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Completion contract must be a JSON object, got {type(parsed)}. "
            f"Content: {content[:500]}..."
        )
    
    if "completed_capability" not in parsed:
        raise ValueError(
            f"Completion contract missing 'completed_capability' field. "
            f"Content: {content[:500]}..."
        )
    
    return parsed
