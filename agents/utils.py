"""
Shared utilities for agent completion contracts.

All agents must use the unified completion contract to signal capability completion.
"""

import json
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage


def build_completion_message(capability: str, data: Optional[Dict[str, Any]] = None) -> AIMessage:
    """
    Build a standardized completion message following the agent completion contract.
    
    Args:
        capability: The capability name that was completed (e.g., "research", "send_email")
        data: Optional payload relevant to the next agent (e.g., file_path, research_summary)
    
    Returns:
        AIMessage with JSON-formatted completion contract
    
    Contract format:
    {
        "completed_capability": "<capability_name>",
        "data": <optional payload>
    }
    """
    contract = {
        "completed_capability": capability,
    }
    
    if data is not None:
        contract["data"] = data
    
    # Serialize to JSON string
    json_content = json.dumps(contract, indent=2)
    
    return AIMessage(content=json_content)


def parse_completion_message(message_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a completion message and extract the completion contract.
    
    Args:
        message_content: The content of an AIMessage
    
    Returns:
        Dict with "completed_capability" and optionally "data", or None if not a valid contract
    """
    if not message_content:
        return None
    
    try:
        # Try to parse as JSON
        parsed = json.loads(message_content)
        
        # Validate contract structure
        if isinstance(parsed, dict) and "completed_capability" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON or not a completion contract
        pass
    
    return None


def extract_completed_capability(message_content: str) -> Optional[str]:
    """
    Extract the completed capability from a message.
    
    Args:
        message_content: The content of an AIMessage
    
    Returns:
        Capability name if found, None otherwise
    """
    contract = parse_completion_message(message_content)
    if contract:
        return contract.get("completed_capability")
    return None
