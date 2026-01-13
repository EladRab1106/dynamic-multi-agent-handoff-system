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
from typing import List, Optional, Dict, Any


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

    for file_path in attachments or []:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        filename = os.path.basename(file_path)
        part = MIMEBase("application", "octet-stream")

        with open(file_path, "rb") as f:
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )

        msg.attach(part)

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
