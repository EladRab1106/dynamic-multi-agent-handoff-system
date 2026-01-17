"""
Utility functions for Direct Answer agent.

Local copy of contract parsing utilities to ensure the agent is fully self-contained.
"""

import json
from typing import Dict, Any, Optional


def parse_completion_message(message_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a completion contract from message content.
    
    Returns:
        Dict with completion contract if valid, None otherwise
    """
    if not message_content:
        return None

    try:
        parsed = json.loads(message_content)
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
