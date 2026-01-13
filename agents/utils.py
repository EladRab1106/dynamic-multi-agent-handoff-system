import json
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage


def build_completion_message(capability: str, data: Optional[Dict[str, Any]] = None) -> AIMessage:
    contract = {
        "completed_capability": capability,
    }

    if data is not None:
        contract["data"] = data

    json_content = json.dumps(contract, indent=2)
    return AIMessage(content=json_content)


def parse_completion_message(message_content: str) -> Optional[Dict[str, Any]]:
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
    contract = parse_completion_message(message_content)
    if contract:
        return contract.get("completed_capability")
    return None
