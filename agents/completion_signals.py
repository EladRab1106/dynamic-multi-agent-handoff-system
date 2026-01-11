"""
Completion signal protocol for agent-to-supervisor communication.

All agents must emit explicit completion signals when their capability is fully completed.
The Supervisor ONLY advances steps when it detects these explicit signals.

Completion signals are deterministic, machine-readable markers that agents include
in their final AIMessage content.
"""

# Completion signal constants
RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
CREATE_DOCUMENT_COMPLETED = "CREATE_DOCUMENT_COMPLETED"
SEND_EMAIL_COMPLETED = "SEND_EMAIL_COMPLETED"
SEARCH_EMAIL_COMPLETED = "SEARCH_EMAIL_COMPLETED"
DIRECT_ANSWER_COMPLETED = "DIRECT_ANSWER_COMPLETED"

# Mapping from signals to capabilities
SIGNAL_TO_CAPABILITY = {
    RESEARCH_COMPLETED: "research",
    CREATE_DOCUMENT_COMPLETED: "create_document",
    SEND_EMAIL_COMPLETED: "send_email",
    SEARCH_EMAIL_COMPLETED: "search_email",
    DIRECT_ANSWER_COMPLETED: "direct_answer",
}

# All valid completion signals
ALL_SIGNALS = set(SIGNAL_TO_CAPABILITY.keys())


def extract_completion_signal(message_content: str) -> str | None:
    """
    Extract completion signal from message content.
    
    Returns the signal string if found, None otherwise.
    Signals must appear as exact matches (case-sensitive).
    """
    if not message_content:
        return None
    
    # Check for exact signal matches
    for signal in ALL_SIGNALS:
        if signal in message_content:
            return signal
    
    return None


def get_capability_from_signal(signal: str) -> str | None:
    """Get capability name from completion signal."""
    return SIGNAL_TO_CAPABILITY.get(signal)


def is_completion_signal(message_content: str) -> bool:
    """Check if message contains a completion signal."""
    return extract_completion_signal(message_content) is not None
