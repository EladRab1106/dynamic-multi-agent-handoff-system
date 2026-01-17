"""
Gmail send tool - sends emails with optional attachments.
"""

from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from base64 import urlsafe_b64encode
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
]


def load_gmail_credentials():
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    if not os.path.exists(token_path):
        raise RuntimeError(
            "Gmail token.json not found. "
            "Run gmail_auth.py to generate OAuth credentials."
        )
    return Credentials.from_authorized_user_file(token_path, SCOPES)


@tool
def gmail_send(
    to: str,
    subject: str,
    body: str,
    attachments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Send an email via Gmail.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body text
        attachments: Optional list of file paths to attach
    
    Returns:
        Dict with status: {"status": "sent", "to": "...", "subject": "..."}
    """
    creds = load_gmail_credentials()
    service = build("gmail", "v1", credentials=creds)

    msg = MIMEMultipart()
    msg["to"] = to
    msg["subject"] = subject
    msg["from"] = os.getenv("GMAIL_SENDER_ADDRESS", to)

    msg.attach(MIMEText(body, "plain"))

    # Log attachments argument
    logger.info(f"gmail_send: Received attachments argument: {attachments}")
    
    # Process attachments with path normalization and validation
    for file_path in attachments or []:
        # Normalize path: expand user home directory and resolve to absolute
        resolved_path = Path(file_path).expanduser().resolve()
        resolved_file_path = resolved_path.as_posix()
        
        logger.info(f"gmail_send: Processing attachment - original: {file_path}, resolved: {resolved_file_path}")
        
        # HARD VALIDATION: File must exist BEFORE sending
        if not resolved_path.exists():
            error_msg = f"Attachment file does not exist: {resolved_file_path} (original: {file_path})"
            logger.error(f"gmail_send: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Log file size BEFORE sending
        file_size = resolved_path.stat().st_size
        logger.info(f"gmail_send: Attachment file exists - path: {resolved_file_path}, size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = f"Attachment file is empty: {resolved_file_path}"
            logger.error(f"gmail_send: {error_msg}")
            raise ValueError(error_msg)
        
        # Use resolved path for reading
        filename = resolved_path.name
        part = MIMEBase("application", "octet-stream")

        with open(resolved_file_path, "rb") as f:
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )

        msg.attach(part)
        logger.info(f"gmail_send: Successfully attached file: {filename} ({file_size} bytes)")

    raw = {
        "raw": urlsafe_b64encode(msg.as_bytes()).decode()
    }

    service.users().messages().send(
        userId="me",
        body=raw,
    ).execute()

    return {
        "status": "sent",
        "to": to,
        "subject": subject
    }
