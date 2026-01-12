RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
CREATE_DOCUMENT_COMPLETED = "CREATE_DOCUMENT_COMPLETED"
SEND_EMAIL_COMPLETED = "SEND_EMAIL_COMPLETED"
SEARCH_EMAIL_COMPLETED = "SEARCH_EMAIL_COMPLETED"
DIRECT_ANSWER_COMPLETED = "DIRECT_ANSWER_COMPLETED"

SIGNAL_TO_CAPABILITY = {
    RESEARCH_COMPLETED: "research",
    CREATE_DOCUMENT_COMPLETED: "create_document",
    SEND_EMAIL_COMPLETED: "send_email",
    SEARCH_EMAIL_COMPLETED: "search_email",
    DIRECT_ANSWER_COMPLETED: "direct_answer",
}

ALL_SIGNALS = set(SIGNAL_TO_CAPABILITY.keys())


def extract_completion_signal(message_content: str) -> str | None:
    if not message_content:
        return None
    
    for signal in ALL_SIGNALS:
        if signal in message_content:
            return signal
    
    return None


def get_capability_from_signal(signal: str) -> str | None:
    return SIGNAL_TO_CAPABILITY.get(signal)


def is_completion_signal(message_content: str) -> bool:
    return extract_completion_signal(message_content) is not None
