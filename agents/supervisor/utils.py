"""
Utility functions for Supervisor agent.

Local copy to ensure the agent is fully self-contained.
"""

import json
from typing import Dict, Any, Optional


def parse_completion_message(message_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a completion contract from message content.
    
    Tries to extract JSON even if embedded in text.
    Returns the contract if valid JSON with completed_capability exists.
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
    Extract the completed_capability from a completion contract message.
    
    Returns the capability string if found, None otherwise.
    """
    contract = parse_completion_message(message_content)
    if contract:
        return contract.get("completed_capability")
    return None
